import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from collections import deque
from dataclasses import dataclass
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import draccus
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    AutoConfig, 
    AutoImageProcessor,
    AutoTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Gensim
import prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 as a2d_cfg

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ActionDecoder(torch.nn.Module):
    def __init__(
        self,
        n_layers=1,
        vis_dim=4096,
        n_joints=16,
        window_size=30,
        hidden_dim=512,
        with_proprio=False,
        ):
        super().__init__()
        
        if with_proprio:
            self.proprio_proj = nn.Linear(n_joints, hidden_dim)
        
        self.latent_action_pool = MAPBlock(
            n_layers=n_layers,
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )
        
        self.visual_pool = MAPBlock(
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 8), 
            nn.GELU(),
            nn.Linear(hidden_dim * 8, n_joints * window_size),
        )
        
    def forward(self, latent_action_tokens, visual_embed, proprio=None):
        
        visual_embed = self.visual_pool(visual_embed)
        
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        
        if proprio is not None:
            proprio = proprio.squeeze(1)
            proprio = self.proprio_proj(proprio)
            action = self.proj(torch.cat((action_token, proprio), dim=1))
        else:
            action = self.proj(action_token)

        return action
    
    
class Wrapped_Model(torch.nn.Module):
    def __init__(
        self,
        vla,
        freeze_vla=True,
        window_size=30,
        decoder_n_layers=1,
        decoder_hidden_dim=512,
        with_proprio=False,
        ):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(
            n_layers=decoder_n_layers,
            hidden_dim=decoder_hidden_dim,
            with_proprio=with_proprio,
            )

        self.decoupled_loss = False
        self.with_proprio = with_proprio
        if freeze_vla:
            self.vla.requires_grad_(False)
            

    def forward(self, batch):
        slow_output = self.slow_forward(batch)
        loss, loss_one_step, latent_action_tokens = self.fast_forward(batch, slow_output)

        return slow_output, loss, loss_one_step, latent_action_tokens


    def slow_forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        return output
    

    def fast_forward(self, batch, slow_output):
        # Task and action latents
        visual_embed = slow_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches*3].to(torch.float)

        latent_tokens = slow_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches*3 : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        # Run specialist policy
        if self.with_proprio:
            proprio = batch['proprio']
        else:
            proprio = None
            
        pred_action = self.action_decoder(latent_action_tokens, visual_embed, proprio).reshape(-1, self.window_size, 16)

        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens



@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "ckpts/univla-7b"                         # Path to univla model ckpt
    lam_path: str = "ckpts/univla-latent-action-model/lam-stage-2.ckpt"
    # Directory Paths
    data_root_dir: str = "./dataset/Manipulation-SimData"           # Path to dataset
    meta_json_dir: str = ""  
    dataset_name: str = "genie_dataset/dustbin"                     # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                             # Fine-tuning batch size
    max_steps: int = 40000                                          # Max number of fine-tuning steps
    save_steps: int = 40000                                         # Interval for checkpoint saving
    learning_rate: float = 1.5e-4                                   # Fine-tuning learning rate
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = False                                         # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_00                               # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    # continually overwrite the latest checkpoint
                                                                    # (If False, saves all checkpoints)
    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 30
    lam_loss_weight: float = 1
        
    freeze_vla: bool = False
    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "univla-geniesim"                          # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    run_id_note: Optional[str] = None     
    
    debug: bool = False
    
    # Decoder
    decoder_n_layers: int = 1
    decoder_hidden_dim: int = 512
    
    with_proprio: bool = False


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:

    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir, cfg.adapter_tmp_dir
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    dual_system = Wrapped_Model(
        vla = vla,
        freeze_vla=cfg.freeze_vla,
        window_size=cfg.window_size,
        decoder_n_layers=cfg.decoder_n_layers,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        with_proprio=cfg.with_proprio,
        ).to(device_id)

    trainable_total_params = sum(p.numel() for p in dual_system.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in dual_system.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 8 * 0.8), gamma=0.1)

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    
    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()
    
    # Load gensim dataset
    from prismatic.vla.datasets import A2dDataset
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=cfg.meta_json_dir,
        data_root_dir=cfg.data_root_dir,
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    text_tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-2B",
        trust_remote_code=True,
        add_eos_token=False,
    )

    text_tokenizer.model_max_length = 4096

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
        shuffle=True,
        statistic=False, #TODO: seems useless
        debug_one_episode=cfg.debug,
        # debug_one_episode=False,
    )

    collator = PaddedCollatorForActionPrediction_Gensim()
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=True,
        collate_fn=collator,
        pin_memory=False,
        num_workers=64,
    )
    
    dual_system, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        dual_system, latent_action_model, optimizer, scheduler, dataloader
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process and not cfg.debug:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        dual_system.train()
        optimizer.zero_grad()
        current_step = 0
        if distributed_state.is_main_process:
            # 创建一个 SummaryWriter 实例
            writer = SummaryWriter(log_dir=cfg.run_root_dir)
                
        for e in range(10000):
            progress.set_description("Epoch " + str(e+1))
                
            for batch_idx, batch in enumerate(dataloader):
                batch["init_pixel_values"] = batch["init_pixel_values"].to(device_id) # [8, 3, 224, 224]
                batch["goal_pixel_values"] = batch["goal_pixel_values"].to(device_id) # [8, 3, 224, 224]
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id) # [8, 6, 224, 224]
                batch['actions'] = batch['actions'].to(device_id) # [8, 12, 7]
                batch['proprio'] = batch['proprio'].to(device_id) # [8, 7]

                if len(batch["hist_init_pixel_values"]) > 1:
                    batch["hist_init_pixel_values"] = batch["hist_init_pixel_values"].to(device_id) # [2, 3, 224, 224]
                    batch["hist_goal_pixel_values"] = batch["hist_goal_pixel_values"].to(device_id) # [2, 3, 224, 224]

                    with torch.no_grad():
                        video = torch.stack([batch["init_pixel_values"], batch["goal_pixel_values"]], dim=1) # [8, 2, 3, 224, 224]
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze() # [8, 4]
                        video = torch.stack([batch["hist_init_pixel_values"], batch["hist_goal_pixel_values"]], dim=1) # [2, 2, 3, 224, 224]
                        latent_action_idx_history = latent_action_model.module.vq_encode(video)['indices'].squeeze() # [2, 4]

                    input_ids_list = []
                    labels_list = []
                    hist_idx = 0
                    
                    if batch['actions'].shape[0] == 1:
                        latent_action_idx_batch = latent_action_idx_batch.unsqueeze(0)
                    
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action
                        
                        if batch['with_hist'][idx]:
                            action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx_history[hist_idx]]

                            hist_action_tokens = ''
                            for i, action in enumerate(action_vocab):
                                hist_action_tokens += action

                            input_prompt = f"What action should the robot take to {batch['instructions'][idx]}? History action " + hist_action_tokens
                            hist_idx += 1
                        else:
                            input_prompt = f"What action should the robot take to {batch['instructions'][idx]}?"

                        # print(input_prompt)
                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": input_prompt},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)
                
                else:
                    with torch.no_grad():
                        video = torch.stack([batch["init_pixel_values"], batch["goal_pixel_values"]], dim=1)
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze()

                    input_ids_list = []
                    labels_list = []
                    
                    if batch['actions'].shape[0] == 1:
                        latent_action_idx_batch = latent_action_idx_batch.unsqueeze(0)
                        
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]

                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action

                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": f"What action should the robot take to {batch['instructions'][idx]}?"},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)

            
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

                # Truncate (if necessary)
                input_ids, labels = input_ids[:, : processor.tokenizer.model_max_length], labels[:, : processor.tokenizer.model_max_length]

                # Get `attention_mask` by checking for `pad_token_id`
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels

                output, act_loss, loss_one_step, latent_action_proj = dual_system(batch)
                loss = act_loss if cfg.freeze_vla else act_loss + (output.loss) * cfg.lam_loss_weight
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(dual_system.parameters(), max_norm=0.3)
                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, dual_system.module.vla.vision_backbone.featurizer.patch_embed.num_patches*3 : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()


                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

                # Push Metrics to W&B (every 10 gradient steps)
                # if distributed_state.is_main_process and gradient_step_idx % 5 == 0 and not cfg.debug:
                    
                #     wandb.log(
                #         {
                #             "train_loss": smoothened_loss,
                #             "latent_action_accuracy": smoothened_action_accuracy,
                #             "action_loss": act_loss.item(),
                #             "action_loss_1step": loss_one_step.item(),
                #             "lr": optimizer.state_dict()['param_groups'][0]['lr']
                #             # "latent_align_loss": latent_align_loss.item(),
                #         },
                #         step=gradient_step_idx + current_step,
                #     )

                # Initialize Logging =>> TensorBoard
                if distributed_state.is_main_process:
                    # 使用 add_scalar 方法记录日志
                    writer.add_scalar('train_loss', smoothened_loss, gradient_step_idx + current_step)
                    writer.add_scalar('latent_action_accuracy', smoothened_action_accuracy, gradient_step_idx + current_step)
                    writer.add_scalar('action_loss', act_loss.item(), gradient_step_idx + current_step)
                    writer.add_scalar('action_loss_1step', loss_one_step.item(), gradient_step_idx + current_step)
                    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], gradient_step_idx + current_step)
                    # writer.add_scalar('latent_align_loss', latent_align_loss.item(), gradient_step_idx + current_step)
                    
                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if (gradient_step_idx + current_step) > 0 and (gradient_step_idx + current_step) % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        if not cfg.freeze_vla:
                            processor.save_pretrained(run_dir)
                            dual_system.module.vla.save_pretrained(save_dir)

                        # Save low-level policy
                        torch.save(dual_system.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder.pt')

                    # Wait for processor and adapter weights to be saved by main process
                    dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(run_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_ckpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                    # Block on Main Process Checkpointing
                    dist.barrier()

            current_step += gradient_step_idx
            # Stop training when max_steps is reached
            if current_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
            
        # 别忘了在适当的时候关闭 writer
        writer.close()

if __name__ == "__main__":
    finetune()