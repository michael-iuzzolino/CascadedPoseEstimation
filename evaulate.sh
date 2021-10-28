#!/bin/bash

CFG_ROOT="experiments/mpii/hourglass"
cfg_list=(
# PUGET
  "${CFG_ROOT}/hourglass_4__td_0_95__distillation__alpha_0_25.yaml"
)
for cfg in "${cfg_list[@]}"
do
    cmd=( python pose_estimation/valid.py )   # create array with one element
    cmd+=( --cfg $cfg )
    cmd+=( --force_overwrite )
    cmd+=( --load_best_ckpt )
#     Run command
    "${cmd[@]}"
done