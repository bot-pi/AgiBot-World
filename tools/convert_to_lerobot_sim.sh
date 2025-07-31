#!/bin/bash
# task names as argument
# example usage:
# ./convert_to_lerobot_sim.sh task_1 task_2 task_3 ... 

for task in "$@"; do
  python tools/convert_to_lerobot_sim.py --src_path dataset/Manipulation-SimData --task_name "$task" --tgt_path test --push_to_hub
done