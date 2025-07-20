""" 
This project is built upon the open-source project 🤗 LeRobot: https://github.com/huggingface/lerobot 

We are grateful to the LeRobot team for their outstanding work and their contributions to the community. 

If you find this project useful, please also consider supporting and exploring LeRobot. 

This script converts Simulated Data in the AgiBot world chanllenge to the LeRobot format, 
which is compatible with the LeRobot simulator.

Lerobot 0.6.0 is required to run this script.
Note Depth image is not supported in the current version of LeRobot, therefore workarounds are used here.

Usage:
python tools/convert_to_lerobot_sim.py \
    --src_path sample/Manipulation-SimData \
    --task_name stamp_the_seal \
    --tgt_path ./test \
    --push_to_hub

"""

import os
import json
import shutil
import logging
import argparse
import gc
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy

import h5py
import torch
import torchvision
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pformat
from tqdm.contrib.concurrent import process_map
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import (
    aggregate_stats, 
    auto_downsample_height_width, 
    get_feature_stats, 
    sample_indices,
)
from lerobot.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
    validate_episode_buffer,
    validate_frame,
    load_image_as_numpy,
    write_info,
    write_episode,
    write_episode_stats,
)

REPO_NAME = "yangyangfu/agibot-simdata-sample"  # Name of the output dataset, also used for the Hugging Face Hub


HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HEAD_DEPTH = "head_depth.png"
HAND_LEFT_DEPTH = "hand_left_depth.png"
HAND_RIGHT_DEPTH = "hand_right_depth.png"


FEATURES = {
    "observation.images.head_color": {
        "dtype": "video",
        "shape": (720, 1280, 3),
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left_color": {
        "dtype": "video",
        "shape": (480, 848, 3),
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right_color": {
        "dtype": "video",
        "shape": (480, 848, 3),
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_depth": {
        "dtype": "float32",
        "shape": (720, 1280, 1),
        #"names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "name": {
            "left_arm_joint": [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
                "position_7",
            ],
            "right_arm_joint": [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
                "position_7",
            ],
            "left_gripper": ["range"],
            "right_gripper": ["range"],
            "head": ["yaw", "pitch"],
            "waist": ["pitch", "lift"],
        },
        "shape": (20,),
    },
    "action": {
        "dtype": "float32",
        "name": {
            "left_arm_joint": [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
                "position_7",
            ],
            "right_arm_joint": [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
                "position_7",
            ],
            "left_gripper": ["range"],
            "right_gripper": ["range"],
            "head": ["yaw", "pitch"],
            "waist": ["pitch", "lift"],
        },
        "shape": (20,),
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,  
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}

# Modified from lerobot.compute_stats.sample_images to deal with video files
def sample_images(path: str | list[str]) -> np.ndarray:
    # assume path is a video file
    if type(path) is str and path.endswith(('.mp4', '.avi', '.mkv')):
        video_path = path
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]

        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img

    # assume image as an array (L, H, W, C): 
    elif type(path) is np.ndarray:
        sampled_indices = sample_indices(len(path))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = path[idx]
            img = auto_downsample_height_width(img)
            
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape)).astype(img.dtype)

            images[i] = img
    
    return images

# Modified from lerobot.compute_stats.compute_episode_stats to handle depth images
def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        # depth image as a float32 3d array
        # (h, w, c)
        elif 'depth' in key:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 1, 2)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255., axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats

# Modified from LerobotDatsetMeta to handle action_config information
class AgiBotDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        action_config: list[dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            "action_config": action_config,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class AgiBotDataset(LeRobotDataset):
    def save_episode(self, 
                    episode_data: dict | None = None, 
                    videos: dict[str, str] | None = None,
                    action_config: dict | None = None
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # copy videos
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        # calculate episode stats in consolidate()
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        
        # to parquet files
        self._save_episode_table(episode_buffer, episode_index)
        
        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, action_config)

        # check timestamps
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        
        # Reset the buffer
        if not episode_data:  
            self.episode_buffer = self.create_episode_buffer()
            
    def add_frame(self, frame: dict, task: str) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1
                    
 
def load_depth(root_dir: str, frame_idx: int, camera_name: str):
    cam_path = Path(root_dir)/"camera"/str(frame_idx)/camera_name
    img = np.array(Image.open(cam_path)).astype(np.float32)/1000. # convert to meters
    if img.ndim == 2:  # if it's a single channel image
        img = img[:, :, np.newaxis]
        
    return img


def load_local_dataset(episode_id: int, src_path: str, save_depth: bool = False) -> list | None:
    """Load local dataset and return a dict with observations and actions"""

    #ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    ob_dir = Path(src_path) / episode_id
    
    with h5py.File(ob_dir / "aligned_joints.h5") as f:
        state_joint = np.array(f["state/joint/position"]) # (14, ) left 1-7, right 8-14
        state_left_effector = np.array(f["state/left_effector/position"]) # (1, )
        state_right_effector = np.array(f["state/right_effector/position"]) # (1, )
        state_head = np.array(f["state/head/position"]) # (2, ) yaw-1, pitch-2
        state_waist = np.array(f["state/waist/position"]) # (2, ) pitch 1, lift 2
        
        action_joint = np.array(f["action/joint/position"])
        action_left_effector = np.array(f["action/left_effector/position"])
        action_right_effector = np.array(f["action/right_effector/position"])
        action_head = np.array(f["action/head/position"])
        action_waist = np.array(f["action/waist/position"])

    states_value = np.hstack(
        [state_joint, state_left_effector, state_right_effector, state_head, state_waist]
    ).astype(np.float32)
    assert (
        action_joint.shape[0] == action_left_effector.shape[0]
    ), f"shape of action_joint:{action_joint.shape};shape of action_left_effector:{action_left_effector.shape}"
    action_value = np.hstack(
        [action_joint, action_left_effector, action_right_effector, action_head, action_waist]
    ).astype(np.float32)

    assert len(states_value) == len(action_value), \
        f"states and actions are not equal in length: {len(states_value)} vs {len(action_value)}"
    
    # load depth images
    #assert len(depth_imgs) == len(
    #    states_value
    #), f"Number of images and states are not equal"
    #assert len(depth_imgs) == len(
    #    action_value
    #), f"Number of images and actions are not equal"
    
    # load rgb images: pass as we will use videos directly
    
    
    # done status
    done = np.zeros((len(states_value),1), dtype=bool)
    done[-1][:] = True
    
    # add frame
    frames = [
        {   
            "observation.state": states_value[i],
            "action": action_value[i],
            "next.done": done[i],
        }
        for i in range(len(states_value))
    ]

    if save_depth:
        for i in range(len(states_value)):
            frames[i]["observation.images.head_depth"] = load_depth(ob_dir, i, HEAD_DEPTH)

    videos = {
        "observation.images.head_color": ob_dir / HEAD_COLOR,
        "observation.images.hand_left_color": ob_dir / HAND_LEFT_COLOR,
        "observation.images.hand_right_color": ob_dir / HAND_RIGHT_COLOR,
    }
    
    print(f"Loaded {len(frames)} frames from episode {episode_id} in {src_path}.")
    return frames, videos

def push_to_subfolder(dataset, task_name):
    """
    Push the dataset to a subfolder in the Hugging Face Hub.
    """
    from huggingface_hub import upload_folder
    
    upload_folder(
        folder_path=dataset.root,
        path_in_repo=task_name,
        repo_id=REPO_NAME,
        repo_type='dataset',
    )
    
       
def main(
    src_path: str,
    tgt_path: str,
    task_name: str,
    repo_id: str,
    task_info_json: str,
    debug: bool = False,
    chunk_size: int = 10,  # Add chunk size parameter
    save_depth: bool = False,  # Add save depth parameter
    push_to_hub: bool = True,  # Add push to hub parameter
):
    with open(task_info_json, "r") as f:
        task_info = json.load(f)
    
    fps = 30
    robot_type = "a2d"  
    use_videos = True  # Use videos instead of images
    
    if not save_depth:
        # Remove depth features if not saving depth
        FEATURES.pop("observation.images.head_depth", None)
    

    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=f"{tgt_path}/{repo_id}/{task_name}",
        robot_type=robot_type,
        use_videos=use_videos,
    )
    
    # overwrite metadata class
    dataset.meta = AgiBotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            features=FEATURES,
            root=f"{tgt_path}/{repo_id}/{task_name}",
            robot_type=robot_type,
            use_videos=use_videos,
        )


    # get all episode id. 
    # The json file contains episode ids that may not be in the src path.
    # Thus we scan the path to get all episode ids.
    num_episodes = len(task_info)
    valid_num_episodes = 0
    all_episode_dir = []
    all_episode_desc = []
    all_action_config = []
    for episode in task_info:
        episode_dir = os.path.join(
            str(episode["task_id"]),
            str(episode["job_id"]),
            str(episode["sn_code"]),
            str(episode["episode_id"]),
        )
        # check if the episode directory exists
        episode_path = Path(src_path) / episode_dir
        if not episode_path.exists():
            logging.warning(f"Episode directory {episode_dir} does not exist in {src_path}. Skipping.")
            continue
        all_episode_dir.append(episode_dir)
        all_episode_desc.append(
            f"{episode['english_task_name']}"
        )
        
        # action config 
        action_config = episode['label_info'].get('action_config', {})
        all_action_config.append(action_config)

        valid_num_episodes += 1
    
    print(f"{valid_num_episodes} valid episodes out of {num_episodes} found in {src_path}.")
    if valid_num_episodes == 0:
        logging.warning(f"No valid episodes found in {src_path}.")
        return
    
    # Debug mode
    if debug:
        all_episode_dir = all_episode_dir[:2]


    # Process in chunks to reduce memory usage
    for chunk_start in tqdm(range(0, len(all_episode_dir), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(all_episode_dir))
        chunk_eids = all_episode_dir[chunk_start:chunk_end]
        chunk_descs = all_episode_desc[chunk_start:chunk_end]
        chunk_action_configs = all_action_config[chunk_start:chunk_end]
        
        # Process only this chunk
        if debug:
            raw_datasets_chunk = [
                load_local_dataset(subdir, src_path=src_path, save_depth=save_depth)
                for subdir in tqdm(chunk_eids, desc="Loading chunk data")
            ]
        else:
            raw_datasets_chunk = process_map(
                partial(load_local_dataset, src_path=src_path, save_depth=save_depth),
                chunk_eids,
                max_workers=os.cpu_count() // 2,
                desc=f"Loading chunk {chunk_start//chunk_size + 1}/{(len(all_episode_desc) + chunk_size - 1)//chunk_size}",
            )
            
        # Filter out None results
        valid_datasets = [(ds, desc, action_config) for ds, desc, action_config in zip(raw_datasets_chunk, chunk_descs, chunk_action_configs) if ds is not None]
        
        # Process each dataset in the chunk
        for raw_dataset, episode_desc, action_config in tqdm(valid_datasets, desc="Processing episodes in chunk"):
            for frame in tqdm(
                raw_dataset[0], desc="Processing frames", leave=False
            ):

                dataset.add_frame(frame=frame, task=episode_desc)
            #dataset.save_episode(task=episode_desc, videos=raw_dataset[1])
            dataset.save_episode(videos=raw_dataset[1], action_config=action_config)
            
        # Clear memory after each chunk
        raw_datasets_chunk = None
        valid_datasets = None
        gc.collect()
    
    # push to hub
    if push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        push_to_subfolder(dataset, task_name)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="Number of episodes to process at once",
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        help="Save depth images as well",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the dataset to the Hugging Face Hub",
    )
    args = parser.parse_args()

    src_path = f"{args.src_path}/{args.task_name}"
    json_file = f"{args.src_path}/{args.task_name}/task_train.json"
    #dataset_base = f"agibotworld/task_{args.task_id}"
    repo_id = REPO_NAME

    assert Path(json_file).exists, f"Cannot find {json_file}."
    main(src_path, args.tgt_path, args.task_name, repo_id, json_file, args.debug, args.chunk_size, args.save_depth, args.push_to_hub)