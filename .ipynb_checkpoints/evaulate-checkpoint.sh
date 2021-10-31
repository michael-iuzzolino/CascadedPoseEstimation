#!/bin/bash

CFG_ROOT="experiments/mpii/hourglass"
cfg_list=(
#   "${CFG_ROOT}/hourglass_4__td_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0__double.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__double.yaml"
  "${CFG_ROOT}/hourglass_4__td_0_25__double.yaml"
  "${CFG_ROOT}/hourglass_4__td_0_5__double.yaml"
  "${CFG_ROOT}/hourglass_4__td_0_9__double.yaml"

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