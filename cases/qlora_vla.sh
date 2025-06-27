# Start training with 1 GPUs nproc-per-node
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 1 \
scripts/finetune.py \
--vla_path ckpts/univla-7b \
--lam_path ckpts/univla-latent-action-model/lam-stage-2.ckpt \
--data_root_dir dataset/Manipulation-SimData \
--codebook_size 16 \
--batch_size 4 \
--grad_accumulation_steps 4 \
--max_steps 300000 \
--save_steps 1000 \
--learning_rate 1e-5 \
--run_root_dir runs/agibot \
--adapter_tmp_dir runs/agibot \
--use_lora \
--use_quantization \
--wandb_project qlora_vla \
--with_proprio \
--wogripper \
--task_ids 1