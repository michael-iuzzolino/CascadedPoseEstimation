#!/bin/bash

CFG_ROOT="experiments/mpii/hourglass"
cfg_list=(
#   "${CFG_ROOT}/hourglass_4__td_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_5.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1.yaml"

#   "${CFG_ROOT}/hourglass_4__td_0__double.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25__double.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_5__double.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__double.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__double.yaml"

#   "${CFG_ROOT}/hourglass_4__td_0__distill_td_0.yaml" 
#   "${CFG_ROOT}/hourglass_4__td_0__distill_td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0__distill_td_0_5.yaml" 
#   "${CFG_ROOT}/hourglass_4__td_0__distill_td_0_9.yaml" 
  "${CFG_ROOT}/hourglass_4__td_0__distill_td_1.yaml"

#   "${CFG_ROOT}/hourglass_4__td_0_25__distill_td_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25__distill_td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25__distill_td_0_5.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25__distill_td_0_9.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_25__distill_td_1.yaml"

#   "${CFG_ROOT}/hourglass_4__td_0_5__distill_td_0.yaml"  
#   "${CFG_ROOT}/hourglass_4__td_0_5__distill_td_0_25.yaml"  
#   "${CFG_ROOT}/hourglass_4__td_0_5__distill_td_0_5.yaml"  
#   "${CFG_ROOT}/hourglass_4__td_0_5__distill_td_0_9.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_5__distill_td_1.yaml"


#   "${CFG_ROOT}/hourglass_4__td_0_9__distill_td_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__distill_td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__distill_td_0_5.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__distill_td_0_9.yaml"
#   "${CFG_ROOT}/hourglass_4__td_0_9__distill_td_1.yaml"

#   "${CFG_ROOT}/hourglass_4__td_1__distill_td_0.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__distill_td_0_25.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__distill_td_0_5.yaml"
#   "${CFG_ROOT}/hourglass_4__td_1__distill_td_0_9.yaml"
  "${CFG_ROOT}/hourglass_4__td_1__distill_td_1.yaml"
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