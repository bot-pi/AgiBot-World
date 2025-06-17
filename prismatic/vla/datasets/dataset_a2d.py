import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "InternVL"))
import copy
import json
import re
import random
import time

random.seed(42)
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import logging
logging.set_verbosity_info()

import argparse
from transformers import AutoTokenizer
import prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 as cfg 
from prismatic.vla.datasets.dataset_transforms import PipelineComposer
from prismatic.vla.datasets.dataset_transforms import build_latent_image_transform
from internvl_chat.internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl_chat.internvl.train.dataset import build_transform, dynamic_preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.get_logger("transformers.dataset_jaka" + __name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def concat_pad_data_collator_pi0(features, max_item_length=None, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]

    for i in range(len(features)):
        feat = features[i]

        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        if "position_ids" in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[: feat["position_ids"].shape[0]] = feat["position_ids"]
            feat["position_ids"] = temp_position_ids

    # Handling of all other possible keys.
    for k, v in first.items():
        if (
            k not in ["labels", "label_ids", "pixel_values", "depth_values", "image_flags"]
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

        if k in ["pixel_values", "depth_values", "image_flags", "videos"]:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])

    return batch


def timer(vis=True):
    """Timer."""

    def time_it(func):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = func(*arg, **kwarg)
            e_time = time.time()
            cost = e_time - s_time
            if vis:
                print(f"[{func.__qualname__}] cost time: {cost:.4f}")
                # logger.info("[%s] cost time: %.3f", func.__name__, cost)
            return res

        return inner

    return time_it


class MetaDataset(Dataset):
    def __init__(
        self,
        label_file_dir,
        data_root_dir,
        valid_episode_txt=None,
    ):
        self.label_file_dir = label_file_dir
        self.data_root_dir = data_root_dir
        if valid_episode_txt is None:
            self.valid_episodes = None
            logger.info(f"[DATASET] not Load valid_episode_txt")
        else:
            with open(valid_episode_txt, "r", encoding="utf-8") as fin:
                valid_episodes: List[str] = [line.strip() for line in fin]
            logger.info(f"[DATASET] Load {len(valid_episodes)} valid episode_ids from {valid_episode_txt}")
            self.valid_episodes = set(valid_episodes)

    #NOTE: added task_folder to be compatiable wit simdata
    def get_episode_path(self, episode_info, task_folder):
        task_id = episode_info["task_id"]
        epath = os.path.join(
            self.data_root_dir, task_folder,
            f'{task_id}/{episode_info["job_id"]}/{episode_info["sn_code"]}/{episode_info["episode_id"]}',
        )
        return epath
    #NOTE: added task_folder to be compatiable with simdata
    def read_meta_info(self, episode_info):
        #epath = self.get_episode_path(episode_info)
        epath = episode_info["episode_dir"]
        meta_path = os.path.join(epath, "meta_info.json")
        with open(meta_path, "r") as fid:
            meta_info = json.load(fid)

        return meta_info

    def predefine_invalid_data_check(self, episode_info):
        task_id = str(episode_info["task_id"])
        job_id = str(episode_info["job_id"])
        sn_code = str(episode_info["sn_code"])
        episode_id = str(episode_info["episode_id"])

        if self.valid_episodes is not None and episode_id not in self.valid_episodes:
            return True

        meta_info = self.read_meta_info(episode_info)
        version = meta_info["version"]
        if version < "v0.0.2":
            # v0.0.1 episode's gripper value means action instead of state.
            return True

        return False


class BaseDataset(MetaDataset):
    def __init__(
        self,
        label_file_dir=None,
        data_root_dir=None,
        world_size=None,
        rank_id=None,
        sample_rate=None,
        online_process_mp_cnt=1,
        valid_episode_txt=None,
    ):
        super().__init__(
            label_file_dir=label_file_dir, data_root_dir=data_root_dir, valid_episode_txt=valid_episode_txt
        )
        self.world_size = world_size
        self.rank_id = rank_id
        self.sample_rate = sample_rate
        self.online_process_mp_cnt = online_process_mp_cnt

        self.check_img = False
        self.check_corrupt = False

    @timer()
    def generate_task_infos(
        self,
        dataset_cfg,
        task_episode_processors_cfg,
        task_dataset_processors_cfg,
        task_runtime_processors_cfg,
        shuffle=True,
        statistic=False,
        debug_one_episode=False,
    ):
        self.dataset_cfg = dataset_cfg
        self.task_episode_processors = PipelineComposer(task_episode_processors_cfg)
        self.task_dataset_processors = PipelineComposer(task_dataset_processors_cfg)
        self.task_runtime_processors = PipelineComposer(task_runtime_processors_cfg)


        all_dataset_episode_info = []
        for idx, (task_id, task_config) in enumerate(dataset_cfg.items()):
            #label_file_name = os.path.join(self.label_file_dir, task_config["label_file_name"])
            label_file_name = os.path.join(self.data_root_dir, task_id, task_config["label_file_name"])
            with open(label_file_name, "r") as fid:
                label_list = json.load(fid)
            label_list = self.pack_addition_info(label_list, task_id, task_config)
            all_dataset_episode_info.extend(label_list)
            logger.info(f"label task{task_id} file: {label_file_name}, contains {len(label_list)} episode info.")

        check_results, all_dataset_episode_info = self.episode_common_sanity_check_and_filter(all_dataset_episode_info)
        logger.info(f"sanity check: {check_results}")

        if debug_one_episode == True:
            logger.info(f"DEBUG MODE: Only use one episode!!")
            all_dataset_episode_info = all_dataset_episode_info[:1]

        if statistic == True:
            self.statistic_draw_dataset(all_dataset_episode_info)

        if self.world_size is None:
            sub_data_shard = all_dataset_episode_info
        else:
            random.shuffle(all_dataset_episode_info)  # shuffle before shard
            sub_data_shard = []
            for idx, info in enumerate(all_dataset_episode_info):
                if idx % self.world_size == self.rank_id:
                    sub_data_shard.append(info)

        self.raw_data = self._processor_pipeline_episode(sub_data_shard)
        logger.info(
            f"[rank:{self.rank_id}/worldsize:{self.world_size}] Get {len(self.raw_data)} episode from all {len(all_dataset_episode_info)} episode in {len(dataset_cfg)} dataset!"
        )
        self.data, self.data_infos = self._processor_pipeline_dataset(self.raw_data)
        # logger.info(f"data_infos: {self.data_infos}")

        if self.sample_rate is not None:  # TODO(hxd): sample at the last process is inefficient
            data_sampled = []
            for idx, item in enumerate(self.data):
                if idx % self.sample_rate == 0:
                    data_sampled.append(item)
            del self.data
            self.data = None
            self.data = data_sampled
        logger.info(f"load {len(self.data)} pair data with sampling ratio: {self.sample_rate}")

        dist.barrier()
        original_length = len(self.data)
        shard_num = torch.tensor([original_length], dtype=torch.int64, device=torch.cuda.current_device())
        dist.all_reduce(shard_num, op=dist.ReduceOp.MIN, async_op=False)
        shard_num = shard_num.cpu().item()
        self.data = self.data[:shard_num]

        if shuffle:
            self.shuffle()
            logger.info(f"shuffle self.data")

        logger.info(f"Finally, get {len(self.data)} pair data, original len is {original_length}")

    #NOTE: added task_id to be compatiable with simdata
    def pack_addition_info(self, labels, task_id, task_config):
        for label in labels:
            label["episode_dir"] = self.get_episode_path(label, task_id)
            label["task_specific_cfg"] = task_config
        return labels

    @timer()
    def _processor_pipeline_episode(self, dataset_episode_infos):
        all_episode_infos = []

        if self.online_process_mp_cnt <= 1:
            for ep_info in tqdm(dataset_episode_infos, desc=f"process_episode", mininterval=60):
                try:
                    infos = self.task_episode_processors(ep_info)
                    all_episode_infos.append(infos)
                except Exception as error:
                    logger.error(f"_processor_pipeline_episode met error:{error}, episode dir:{ep_info['episode_dir']}")
        else:
            raise NotImplementedError(f"online_process_mp_cnt > 1 is not implemented.")
            # all_episode_infos = self.task_episode_processors_multiprocess(list(raw_data.keys()), raw_data)

        return all_episode_infos

    @timer()
    def _processor_pipeline_dataset(self, raw_data):
        data_infos = {}
        results = self.task_dataset_processors({"dataset": raw_data, "global_info": data_infos})
        data = results["iter_dataset"]
        return data, data_infos

    @timer()
    def statistic_draw_dataset(self, episode_infos):
        data = []
        for episode_info in episode_infos:
            meta_info_path = os.path.join(episode_info["episode_dir"], "meta_info.json")
            with open(meta_info_path, "r", encoding="utf-8") as file:
                meta_info = json.load(file)
            meta_text = json.loads(meta_info["text"])["description"]

            def extract_items_from_text(text):
                """
                提取文本中大括号内的所有非空字符串，并处理可能存在的转义字符。

                :param text: 要分析的文本字符串
                :return: 包含所有匹配项的列表
                """
                # 正则表达式模式：匹配大括号内的非空字符串，忽略转义的大括号
                pattern = r"(?<!\\)\{(.*?)\}"

                matches = re.findall(pattern, text)

                # 去除转义符后的额外引号
                cleaned_matches = [match.replace('\\"', '"') for match in matches]
                return cleaned_matches

        return

    @timer()
    def episode_common_sanity_check_and_filter(self, episode_infos):
        sanity_check_result = {
            "ep_dir_not_exist": 0,
            "meta_info_not_exist": 0,
            "aligned_state_not_exist": 0,
            "predefined_invalid": 0,
        }

        episode_info_filtered = []
        for ep_info in tqdm(episode_infos, desc="sanity_check", mininterval=60):
            epath = ep_info["episode_dir"]
            if not os.path.exists(epath):
                sanity_check_result["ep_dir_not_exist"] += 1
                continue

            if not os.path.exists(os.path.join(ep_info["episode_dir"], "meta_info.json")):
                sanity_check_result["meta_info_not_exist"] += 1
                continue

            if self.predefine_invalid_data_check(ep_info):
                sanity_check_result["predefined_invalid"] += 1
                continue

            episode_info_filtered.append(ep_info)
        return sanity_check_result, episode_info_filtered

    def __len__(self):
        try:
            return len(self.data)
        except Exception as e:
            raise RuntimeError(f"task dataset may not init!, {e}")

    def __getitem__(self, idx):
        raw_target_ = self.data[idx]
        raw_target = copy.deepcopy(raw_target_)
        raw_target = self.task_runtime_processors(raw_target)
        return raw_target

    def shuffle(self):
        random.shuffle(self.data)


class A2dDataset(BaseDataset):
    def __init__(
        self,
        # internvl language related args
        text_tokenizer,
        # internvl vision related args
        num_image_token,
        is_train=True,
        image_size=448,
        pad2square=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type="imagenet",
        # action expert related args
        action_chunk_size=30,
        use_real_state=False,
        conversation_type=0,
        vis_frame=False,
        vis_dir=None,
        ActionSpacePadder=None,
        min_window_size: int = 16,
        max_window_size: int = 16,
        image_transform = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}")

        self.text_tokenizer = text_tokenizer
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.action_chunk_size = action_chunk_size
        self.use_real_state = use_real_state
        self.conversation_type = conversation_type
        self.vis_frame = vis_frame
        self.vis_dir = vis_dir

        self.ActionSpacePadder = ActionSpacePadder

        self.min_window_size = min_window_size 
        self.max_window_size = max_window_size 

        self.image_transform = image_transform
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.resize_img = torchvision.transforms.Resize(224)

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    @staticmethod
    def make_conversation(conversation_type, prompt_dict):
        if conversation_type == 0:
            return f"What action should the robot take to {prompt_dict['job_description']}?"
        elif conversation_type == 1:
            random_num = random.random()
            if random_num < 0.33:
                return f"What action should the robot take to {prompt_dict['job_description']}?"
            elif random_num < 0.66:
                return f"The robot is performing the step of {prompt_dict['sub_job_description']}."
            else:
                return f"What action should the robot take to {prompt_dict['job_description']}? The robot is performing the step of {prompt_dict['sub_job_description']}."
        else:
            logger.error(f"Conversation Type {conversation_type} is not implemented.")
            raise NotImplementedError()

    def _get_conversation(self, raw_target):
        return A2dDataset.make_conversation(self.conversation_type, raw_target)

    def multi_image_get_item(
        self,
        raw_target: Dict[str, Any],
        cam_keys: List[str] = ["cam_tensor_head_color", "cam_tensor_hand_right_color", "cam_tensor_hand_left_color"],
    ):

        images, num_tiles = [], []
        num_image = 0

        ret = {}
        for cam_key in cam_keys:
            if cam_key in raw_target:
                num_image += 1
                if self.dynamic_image_size:
                    image = dynamic_preprocess(
                        raw_target[cam_key],
                        min_num=self.min_dynamic_patch,
                        max_num=self.max_dynamic_patch,
                        image_size=self.image_size,
                        use_thumbnail=self.use_thumbnail,
                    )
                    images += image
                    num_tiles.append(len(image))
                else:
                    if "hist" in cam_key:
                        ret[cam_key[:10]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))
                    else:
                        if "init" in cam_key:
                            if "head" in cam_key:
                                ret["head_pixel_values"] = self.image_transform(raw_target[cam_key])
                                ret[cam_key[:5]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))   
                            if "left" in cam_key:
                                ret["left_hand_pixel_values"] = self.image_transform(raw_target[cam_key])
                            if "right" in cam_key:
                                ret["right_hand_pixel_values"] = self.image_transform(raw_target[cam_key])                 
                        else:
                            ret[cam_key[:5]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))   
                                     
        ret["pixel_values"] = torch.cat((ret["head_pixel_values"], ret["left_hand_pixel_values"], ret["right_hand_pixel_values"]), dim=0)
        
        return ret
    

    def __getitem__(self, idx):
        get_data_done = False
        while not get_data_done:
            try:
                raw_target = super().__getitem__(idx)
                window_size = raw_target["window_size"]

                action, action_mask = self.ActionSpacePadder.get_action(raw_target["action_target"], chunk_size=30)
                state, state_mask = self.ActionSpacePadder.get_action(raw_target["agent_state"], chunk_size=1)

                results = self.multi_image_get_item(
                    raw_target, 
                    cam_keys=[
                        "init_cam_tensor_head_color", 
                        "init_cam_tensor_hand_right_color", 
                        "init_cam_tensor_hand_left_color", 
                        "goal_cam_tensor_head_color", 
                        "hist_init_cam_tensor_head_color", 
                        "hist_goal_cam_tensor_head_color", 
                        ]
                        )
                freq = int(raw_target["used_cam_cfg"]["head"]["camera_fps"])
                agent_state = torch.tensor(state, dtype=torch.float32)
                if not self.use_real_state:
                    agent_state = -1 * torch.ones_like(agent_state)
                
                results.update(
                    {
                        "actions": torch.tensor(action, dtype=torch.float32),
                        "actions_mask": None,
                        "proprio": agent_state,
                        "ctrl_freqs": torch.tensor([freq], dtype=torch.float32),
                        "window_size": window_size,
                        "lang": raw_target["detailed_job_description"]
                    }
                )
                get_data_done = True
            except Exception as error:
                logger.error(f"process dataset idx: {idx}, {self.data[idx]['episode_dir']}, error info: {error}")
                idx = random.randint(0, len(self.data) - 1)

        return results


class LAMStage1Dataset(BaseDataset):
    def __init__(self, is_train=True, image_size=448, pad2square=False, normalize_type="imagenet", **kwargs):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.normalize_type = normalize_type

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def multi_image_get_item(self, raw_target: Dict[str, Any]):
        img = raw_target["cam_tensor_head_color"]
        img_k = raw_target["cam_tensor_head_color_target"]
        initial_pixel_values = build_latent_image_transform()(img)
        target_pixel_values = build_latent_image_transform()(img_k)
        initial_pixel_values = torch.from_numpy(np.array(initial_pixel_values).astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        target_pixel_values = torch.from_numpy(np.array(target_pixel_values).astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0)

        # Create the final return dictionary
        ret = dict(video=video)
        return ret

    def __getitem__(self, idx):
        get_data_done = False
        while not get_data_done:
            try:
                raw_target = super().__getitem__(idx)
                results = {}
                freq = int(raw_target["used_cam_cfg"]["head"]["camera_fps"])

                results.update(
                    {
                        "random_video_len": raw_target["random_video_len"],
                        "videos": raw_target["videos"],
                        "ctrl_freqs": torch.tensor([freq], dtype=torch.float32),
                    }
                )
                get_data_done = True
            except Exception as error:
                logger.error(f"process dataset idx: {idx}, {self.data[idx]['episode_dir']}, error info: {error}")
                idx = random.randint(0, len(self.data) - 1)

        return results
    
    
def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    # Parse environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the distributed environment
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)  # or 'gloo' for CPU

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Whether using debug mode to dump data into pickle")
    parser.add_argument("--vis_frame", action="store_true", help="Whether visualizing image")
    parser.add_argument("--vis_dir", type=str, default="Data/Images", help="The dictionary to save images")
    parser.add_argument("--video_dir", type=str, default="Data/Videos", help="The dictionary to save videos")
    parser.add_argument("--statistic", action="store_true", help="Whether to statistic data")

    args = parser.parse_args()
    text_tokenizer = AutoTokenizer.from_pretrained(
        "InternVL2-2B",
        trust_remote_code=True,
        add_eos_token=False,
    )

    text_tokenizer.model_max_length = 4096
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = text_tokenizer.add_tokens(token_list, special_tokens=True)

    setup_distributed()
    dataset_args = cfg.DatasetArguments()
    data_training_args = cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = cfg.ActionSpacePadderArguments()

    args.debug = False
    if args.debug:
        ds = BaseDataset(
            label_file_dir=dataset_args.meta_json_dir,
            data_root_dir=dataset_args.data_root_dir,
            world_size=None,
            rank_id=None,
            sample_rate=None,
            online_process_mp_cnt=1,
        )
    else:
        ds = A2dDataset(
            # base parmas
            label_file_dir=dataset_args.meta_json_dir, 
            data_root_dir=dataset_args.data_root_dir, 
            valid_episode_txt=dataset_args.valid_episode_txt, 
            world_size=dist.get_world_size(), 
            rank_id=dist.get_rank(), 
            sample_rate=dataset_args.train_sample_rate, 
            online_process_mp_cnt=dataset_args.online_process_mp_cnt, 
            # a2d params
            text_tokenizer=text_tokenizer, 
            num_image_token=int((dataset_args.force_image_size // 14) ** 2 * (0.5**2)), 
            is_train=True, 
            image_size=data_training_args.force_image_size, 
            pad2square=data_training_args.pad2square, 
            dynamic_image_size=data_training_args.dynamic_image_size, 
            use_thumbnail=data_training_args.use_thumbnail, 
            min_dynamic_patch=data_training_args.min_dynamic_patch, 
            max_dynamic_patch=data_training_args.max_dynamic_patch, 
            normalize_type=data_training_args.normalize_type, 
            action_chunk_size=data_training_args.action_chunk_size, 
            use_real_state=data_training_args.use_real_state, 
            conversation_type=data_training_args.conversation_type, 
            vis_frame=args.vis_frame, 
            vis_dir=args.vis_dir, 
            ActionSpacePadder=ActionSpacePadder, 
        )
    ds.generate_task_infos(
        dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=True,
        statistic=args.statistic,
        debug_one_episode=True,
    )
    dataloader = DataLoader(
        ds,
        batch_size=1, 
        num_workers=12, 
        shuffle=True,
        collate_fn=concat_pad_data_collator_pi0,
    )
    visualized_tensor = []
    for batch_data in tqdm(dataloader):
        data = batch_data
        print(len(batch_data))