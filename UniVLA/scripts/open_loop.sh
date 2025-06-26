# Start training with 1 GPUs nproc-per-node
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 1 \
scripts/open_loop.py \
--pretrained_checkpoint ckpts/univla-7b \
--action_decoder_path runs/agibot/action_decoder.pt \
--data_root_dir dataset/Manipulation-SimData \
--window_size 30 \
--n_layers 1 \
--hidden_dim 512 \
--balancing_factor 0.01 \
--save_path '' \
--task_ids 1 \
--with_proprio \
--debug
