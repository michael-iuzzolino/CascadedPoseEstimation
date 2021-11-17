#!/bin/bash

CFG_ROOT="experiments/mpii"
cfg_list=(
#   "${CFG_ROOT}/hourglass_8__teacher.yaml"
  "${CFG_ROOT}/hourglass_8__td_1__no_shared__distill.yaml"
#   "${CFG_ROOT}/hourglass_8__td_1__no_shared__no_distill.yaml"
#   "${CFG_ROOT}/hourglass_8__td_0_5__shared__distill.yaml"
#   "${CFG_ROOT}/hourglass_8__td_1__no_shared__distill__alpha_0_5.yaml"
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