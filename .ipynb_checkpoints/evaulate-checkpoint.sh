#!/bin/bash

CFG_ROOT="experiments/mpii/hourglass"
cfg_list=(
#   "${CFG_ROOT}/hourglass_4__td_1__distillation__alpha_0_75.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_5__distillation__alpha_0_75.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__distillation__alpha_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__distillation__alpha_0.yaml"
  "${CFG_ROOT}/hourglass_4__td_0_9__distillation__alpha_0_25.yaml"
  "${CFG_ROOT}/hourglass_4__td_0_9__distillation__alpha_0_75.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_5__distillation__alpha_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0__distillation__alpha_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0__distillation__alpha_1.yaml"
)
for cfg in "${cfg_list[@]}"
do
    cmd=( python pose_estimation/valid.py )   # create array with one element
    cmd+=( --cfg $cfg )
    cmd+=( --force_overwrite )
    # Run command
    "${cmd[@]}"
done