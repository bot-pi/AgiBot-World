# Start training with 1 GPUs nproc-per-node
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 1 \
scripts/finetune_genie.py \
--vla_path ckpts/univla-7b \
--lam_path ckpts/univla-latent-action-model/lam-stage-2.ckpt \
--data_root_dir dataset/Manipulation-SimData \
--codebook_size 16 \
--batch_size 1 \
--grad_accumulation_steps 1 \
--max_steps 5000 \
--save_steps 1000 \
--run_root_dir runs/agibot \
--adapter_tmp_dir runs/agibot \
--freeze_vla False \
--use_lora True \
--use_quantization True \
--wandb_project qlora_vla \
--debug False