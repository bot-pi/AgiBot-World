import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.distributed as dist
import tqdm
import argparse
from PIL import Image
from transformers import AutoProcessor
import numpy as np
import matplotlib.pyplot as plt
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
import torch.distributed as dist
import prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 as a2d_cfg
from robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel


def calc_mse_for_single_trajectory(
    policy,
    dataset,
    cfg,
    traj_id: int,
    steps=300,
    action_horizon=30,
    plot=False,
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    length = len(dataset.data)
    steps = min(steps, length)

    for step in tqdm.tqdm(range(steps)):

        data_point = dataset.__getitem__(step)

        # 打开图像文件
        img_h = Image.open(os.path.join(dataset.data[step]["episode_dir"], "camera", str(step), "head_color.jpg"))
        img_l = Image.open(os.path.join(dataset.data[step]["episode_dir"], "camera", str(step), "hand_left_color.jpg"))
        img_r = Image.open(os.path.join(dataset.data[step]["episode_dir"], "camera", str(step), "hand_right_color.jpg"))

        lang = dataset.data[step]["detailed_job_description"]
        
        # 将图像转换为numpy数组
        img_h = np.array(img_h)
        img_l = np.array(img_l)
        img_r = np.array(img_r)

        state = data_point["proprio"][0].numpy()
        gt_action = data_point["actions"][0].numpy()

        state_joints_across_time.append(state)
        gt_action_joints_across_time.append(gt_action)

        # import ipdb;ipdb.set_trace()
        if cfg.with_proprio:
            action = policy.step(img_h, img_l, img_r, lang, state)
        else:
            action = policy.step(img_h, img_l, img_r, lang)
        
        # import ipdb;ipdb.set_trace()
        concat_pred_action = action
        pred_action_joints_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # calc MSE across time
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")

            ax.set_title(f"Joint {i}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(cfg.save_path)

    return mse


def get_policy(cfg):

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    # Load gensim dataset
    from prismatic.vla.datasets import A2dDataset
    train_set = {}
    val_set = {}
    for num in cfg.task_ids:
        train_set[str(num)] = {
            "use_cam_list": ["head", "hand_right", "hand_left"],
            "label_file_name": f"task_{num}_train.json",
        }
        val_set[str(num)] = {
            "use_cam_list": ["head", "hand_right", "hand_left"],
            "label_file_name": f"task_{num}_val.json",
        }
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=cfg.meta_json_dir,
        data_root_dir=cfg.data_root_dir,
        dataset_task_cfg=train_set,
        eval_dataset_task_cfg=val_set,
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

    vla_dataset = A2dDataset(
        # base parmas
        label_file_dir=dataset_args.meta_json_dir, 
        data_root_dir=dataset_args.data_root_dir, 
        valid_episode_txt=dataset_args.valid_episode_txt, 
        world_size=dist.get_world_size(), 
        rank_id=dist.get_rank(), 
        sample_rate=dataset_args.train_sample_rate, 
        online_process_mp_cnt=dataset_args.online_process_mp_cnt, 
        # a2d params
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
        # use_real_state=data_training_args.use_real_state, 
        use_real_state=True, 
        conversation_type=data_training_args.conversation_type, 
        vis_frame=False, 
        vis_dir="", 
        ActionSpacePadder=ActionSpacePadder, 
        min_window_size=cfg.window_size, 
        max_window_size=cfg.window_size + 1, 
        image_transform=processor.image_processor.apply_transform, 
    )

    vla_dataset.generate_task_infos(
        dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=False,
        statistic=False,
        debug_one_episode=True,
    )

    return policy, vla_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenLoop Config")
    
    # argparse
    parser.add_argument("--pretrained_checkpoint", type=str, default="checkpoints/finetuned", help="Path to univla model ckpt")
    parser.add_argument("--action_decoder_path", type=str, default="", help="Path to ADR")
    parser.add_argument("--data_root_dir", type=str, default="", help="Path to dataset")
    parser.add_argument("--meta_json_dir", type=str, default="", help="Path to dataset meta json")
    parser.add_argument("--window_size", type=int, default=30, help="Window size")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--decoder_n_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--decoder_hidden_dim", type=int, default=1024, help="Decoder hidden dimension")
    parser.add_argument("--with_proprio", action="store_true", help="Whether to use proprioceptive data")
    parser.add_argument("--wogripper", action="store_true", help="Whether to not use proprioceptive gripper data")
    parser.add_argument("--save_path", type=str, default="", help="Path to output openloop fig")
    parser.add_argument("--task_ids", nargs='+', type=int, default=[1], help="List of task IDs")
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to 8-bit quantize VLA")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to 4-bit quantize VLA")
    parser.add_argument("--center_crop", action="store_true", help="Whether to sue center crop")
    parser.add_argument("--n_layers", type=int, default=2, help="decoder layers num")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="decoder hidden dim")
    parser.add_argument("--balancing_factor", type=int, default=0.01, help="balancing_factor")

    args = parser.parse_args()

    policy, dataset = get_policy(args)

    mse = calc_mse_for_single_trajectory(
        policy,
        dataset,
        args,
        traj_id=0,
        steps=740,
        action_horizon=30,
        plot=True
    )

    print("MSE loss for trajectory 0:", mse)