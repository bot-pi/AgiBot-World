import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import TrainingArguments

from prismatic.vla.datasets.alpha_base_cfg import BaseDatasetArguments, BaseDataTrainingArguments, BaseModelArguments

RUNNAME = os.environ.get("RUNNAME")


class UniformAction:
    # Right Arm
    RIGHT_ARM_JOINT_POSITIONS = ("right_arm_joint_positions", 7)  # 0:shoulder  6:hand
    RIGHT_END_EFFECTOR_6D_POSE = ("right_end_effector_6d_pose", 6)
    RIGHT_GRIPPER_JOINT_POSITIONS = ("right_gripper_joint_positions", 1)
    RIGHT_DEXTEROUS_HAND_POSITIONS = ("right_dexterous_hand_positions", 6)  # 灵巧手
    RIGHT_ARM_JOINT_VELOCITIES = ("right_arm_joint_velocities", 7)
    RIGHT_GRIPPER_JOINT_VELOCITIES = ("right_gripper_joint_velocities", 5)
    RIGHT_END_EFFECTOR_POSITIONS = ("right_end_effector_positions", 3)
    RIGHT_END_EFFECTOR_VELOCITIES = ("right_end_effector_velocities", 3)
    RIGHT_END_EFFECTOR_ANGULAR_VELOCITIES = ("right_end_effector_angular_velocities", 3)
    # Left Arm
    LEFT_ARM_JOINT_POSITIONS = ("left_arm_joint_positions", 7)
    LEFT_END_EFFECTOR_6D_POSE = ("left_end_effector_6d_pose", 6)
    LEFT_GRIPPER_JOINT_POSITIONS = ("left_gripper_joint_positions", 1)
    LEFT_DEXTEROUS_HAND_POSITIONS = ("left_dexterous_hand_positions", 6)  # 灵巧手
    LEFT_ARM_JOINT_VELOCITIES = ("left_arm_joint_velocities", 7)
    LEFT_GRIPPER_JOINT_VELOCITIES = ("left_gripper_joint_velocities", 5)
    LEFT_END_EFFECTOR_POSITIONS = ("left_end_effector_positions", 3)
    LEFT_END_EFFECTOR_VELOCITIES = ("left_end_effector_velocities", 3)
    LEFT_END_EFFECTOR_ANGULAR_VELOCITIES = ("left_end_effector_angular_velocities", 3)
    # BODY
    HEAD_JOINT_POSITIONS = ("head_joint_positions", 2)
    WAIST_LIFT_POSITIONS = ("waist_lift_positions", 1)
    WAIST_JOINT_POSITIONS = ("waist_joint_positions", 1)
    BASE_LINEAR_VELOCITIES = ("base_linear_velocities", 2)
    BASE_ANGULAR_VELOCITIES = ("base_angular_velocities", 1)

    
class ActionSpacePadder():
    action_space = {}
    action_space_len = 0

    @classmethod
    def get_space(cls, ):
        cls.space_used = cls.space_used_left + cls.space_used_right + cls.space_used_body
        for attribute_name in sorted(dir(UniformAction)):
            if not attribute_name.startswith("__") and attribute_name in cls.space_used:
                attribute_value = getattr(UniformAction, attribute_name)
                cls.action_space.update({attribute_value[0]: (cls.action_space_len, cls.action_space_len + attribute_value[1])})
                cls.action_space_len += attribute_value[1]

    @classmethod
    def get_action(cls, targets, chunk_size=1):
        action = np.zeros((chunk_size, cls.action_space_len), dtype=np.float32)
        mask = np.zeros((1, cls.action_space_len), dtype=np.float32)
        for key, value in targets.items():
            if key in cls.action_space:
                value = np.array(value)
                start_idx, max_idx = cls.action_space[key]
                if value.shape[-1] <= (max_idx - start_idx):
                    action[:, start_idx : start_idx + value.shape[-1]] = value
                    mask[0, start_idx : start_idx + value.shape[-1]] = np.ones_like(value[0])
                else:
                    raise ValueError(
                        f"Invalid action items! {key}:{value}, should follow start_idx:{start_idx}, max_idx:{max_idx}"
                    )
        return action, mask
    
    
class CommonCfg:
    type = "common_cfg"
    force_image_size = 448


class ActionHead:
    type = "action_expert"
    shift = 1
    chunk = 30
    use_real_state = False  # real joint_abs_position or full -1 fake state


train_task_ids = [
    "clear_the_countertop_waste",
    "heat_the_food_in_the_microwave",
    "open_drawer_and_store_items",
    "pack_in_the_supermarket",
    "restock_supermarket_items"
]

train_set = {}
val_set = {}
for num in train_task_ids:
    train_set[str(num)] = {
        "use_cam_list": ["head", "hand_right", "hand_left"],
        "label_file_name": f"task_train.json",
    }
    val_set[str(num)] = {
        "use_cam_list": ["head", "hand_right", "hand_left"],
        "label_file_name": f"task_val.json",
    }


@dataclass
class DatasetArguments(BaseDatasetArguments):
    meta_json_dir: Optional[str] = "dustbin"
    data_root_dir: Optional[str] = "dustbin"
    valid_episode_txt: Optional[str] = None
    use_train_dataset_cache: str = field(default="")
    use_eval_dataset_cache: str = field(default="")
    train_sample_rate: Optional[int] = field(default=None)
    eval_sample_rate: Optional[int] = field(default=400)
    online_process_mp_cnt: int = field(default=1)
    force_image_size: int = field(default=CommonCfg.force_image_size)

    dataset_task_cfg: Optional[Dict[str, str]] = field(default_factory=lambda: train_set)
    episode_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="EpisodeProcessorLoad", gripper_source="state"),  # gripper_source: state or action
            dict(type="EpisodeProcessorJoint2Eef"),
            dict(type="EpisodeProcessorRelableStaticFrames"),
            dict(type="EpisodeProcessorInterpolateGripperValue", downsample_ratio=15, g_max=120),
            dict(type="EpisodeProcessorNormalizeGripperValue", g_min=0, g_max=120, verbose=True),
            # dict(type="EpisodeProcessorVlaBinaryDexterousHand"),
            # dict(type="EpisodeProcessorVlaLabelKeyFrames"),
            # dict(type="EpisodeProcessorVlaStageLabel"),
            # dict(type="EpisodeProcessorRemoveResumePosition", shift=30),
            # dict(type="EpisodeProcessorVlaRemapInstruction"),
        ]
    )
    dataset_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(
                type="DatasetTargetDualArmChunk",
                action_chunk_size=ActionHead.chunk,
                action_shift=ActionHead.shift,
                action_use_delta=False,
                delta_type="frame",
                gripper_use_delta=False,
                mp_cnt=20,
            ),
        ]
    )
    runtime_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="RuntimePromptGeneration", prompt_mode_list=[0, 1, 2, 3]),
            dict(type="RuntimeImagePreprocessLoad"),
            dict(type="RuntimeImageResize", size=(CommonCfg.force_image_size, CommonCfg.force_image_size)),
            dict(
                type="RuntimeActionNorm",
                norm_keys=["left_end_effector_6d_pose", "right_end_effector_6d_pose"],
                params=[166 * 2, 106 * 2, 142 * 2, 27 * 2, 38 * 2, 23 * 2],
                max_value=1,
                min_value=-1,
            ),
            # dict(type="RuntimeImageAugColorJitter", prob_to_process=0.33),
            # dict(type="RuntimeImageAugCorrupt", prob_to_process=0.33),
            # dict(
            #     type="RuntimeImageAugRandomDropImage",
            #     prob_to_process=[0.1, 0.1, 0.1],
            #     images=["cam_tensor_head_color", "cam_tensor_hand_right_color", "cam_tensor_hand_left_color"],
            # ),
        ]
    )

    # Evaluation related configs
    eval_dataset_task_cfg: Optional[Dict[str, str]] = field(default_factory=lambda: val_set)
    eval_episode_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="EpisodeProcessorLoad", gripper_source="state"),  # gripper_source: state or action
            dict(type="EpisodeProcessorJoint2Eef"),
            dict(type="EpisodeProcessorRelableStaticFrames"),
            dict(type="EpisodeProcessorInterpolateGripperValue", downsample_ratio=15, g_max=120),
            dict(type="EpisodeProcessorNormalizeGripperValue", g_min=0, g_max=120, verbose=True),
            # dict(type="EpisodeProcessorVlaBinaryDexterousHand"),
            # dict(type="EpisodeProcessorVlaLabelKeyFrames"),
            # dict(type="EpisodeProcessorVlaStageLabel"),
            # dict(type="EpisodeProcessorRemoveResumePosition", shift=30),
            # dict(type="EpisodeProcessorVlaRemapInstruction"),
        ]
    )
    eval_dataset_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(
                type="DatasetTargetDualArmChunk",
                action_chunk_size=ActionHead.chunk,
                action_shift=ActionHead.shift,
                action_use_delta=False,
                delta_type="frame",
                gripper_use_delta=False,
                mp_cnt=20,
            ),
        ]
    )
    eval_runtime_processors: Optional[List[Dict]] = field(
        default_factory=lambda: [
            dict(type="RuntimePromptGeneration", prompt_mode_list=[0, 1, 2, 3]),
            dict(type="RuntimeImagePreprocessLoad"),
            dict(type="RuntimeImageResize", size=(CommonCfg.force_image_size, CommonCfg.force_image_size)),
            dict(
                type="RuntimeActionNorm",
                norm_keys=["left_end_effector_6d_pose", "right_end_effector_6d_pose"],
                params=[166 * 2, 106 * 2, 142 * 2, 27 * 2, 38 * 2, 23 * 2],
                max_value=1,
                min_value=-1,
            ),
        ]
    )


@dataclass
class ModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="")
    drop_path_rate: float = field(default=0.1)
    freeze_llm: bool = field(default=True)
    freeze_backbone: bool = field(default=True)
    freeze_mlp: bool = field(default=True)
    output_logits: bool = field(default=False)
    grad_checkpoint: bool = field(default=False)


@dataclass
class DataTrainingArguments(BaseDataTrainingArguments):
    max_dynamic_patch: int = field(default=6)
    max_seq_length: int = field(default=4096)
    action_chunk_size: int = field(default=ActionHead.chunk)
    use_real_state: bool = field(default=ActionHead.use_real_state)
    dynamic_image_size: bool = field(default=False)
    pad2square: bool = field(default=False)
    force_image_size: int = field(default=CommonCfg.force_image_size)
    normalize_type: str = field(default="imagenet")
    conversation_type: int = field(default=0)


@dataclass
class AlphaTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=20)
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=100.0)
    per_device_train_batch_size: int = field(default=36)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-4)  # used for training action expert from scratch and freeze VLM
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    do_train: bool = field(default=True)
    group_by_length: bool = field(default=True)
    deepspeed: str = field(default="internvl_chat/zero_stage1_config.json")
    eval_on_start: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=2500)
    per_device_eval_batch_size: int = field(default=16)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=100)
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")
    label_names: List[str] = field(default_factory=lambda: ["action_gts"])
    load_method: str = field(
        default="initialize"
    )  # [initialize,from_pretrained]load param by initialize args, ensuring AE initialize properly


class ActionSpacePadderArguments(ActionSpacePadder):
    space_used_left = [
        "LEFT_ARM_JOINT_POSITIONS",
        # "LEFT_END_EFFECTOR_6D_POSE",
        "LEFT_GRIPPER_JOINT_POSITIONS",
    ]
    space_used_right = [
        "RIGHT_ARM_JOINT_POSITIONS",
        # "RIGHT_END_EFFECTOR_6D_POSE",
        "RIGHT_GRIPPER_JOINT_POSITIONS",
    ]
    space_used_body = [
        # "HEAD_JOINT_POSITIONS",
        # "WAIST_JOINT_POSITIONS",
        # "WAIST_LIFT_POSITIONS",
    ]


ActionSpacePadderArguments.get_space()
