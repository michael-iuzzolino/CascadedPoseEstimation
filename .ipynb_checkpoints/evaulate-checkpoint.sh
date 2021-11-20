#!/bin/bash

CFG_ROOT="experiments/mpii/hourglass"
MAX_BATCH_LOGS=2
DATASET_KEY="test"

THRESHOLD_VALS=( 0.5 )
cfg_list=(
  # Tied Weights, No Distillation
  "${CFG_ROOT}/hourglass_8__td_0.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_25.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_5.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_9.yaml"
  "${CFG_ROOT}/hourglass_8__td_1.yaml"

  # Untied Weights, Distillation
  "${CFG_ROOT}/hourglass_8__td_0__distill_td_1_untied.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_25__distill_td_1_untied.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_5__distill_td_1_untied.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_9__distill_td_1_untied.yaml"
  "${CFG_ROOT}/hourglass_8__td_1__distill_td_1_untied.yaml"

  # Tied Weights, Distillation
  "${CFG_ROOT}/hourglass_8__td_0__distill_td_1.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_25__distill_td_1.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_5__distill_td_1.yaml"
  "${CFG_ROOT}/hourglass_8__td_0_9__distill_td_1.yaml"
  "${CFG_ROOT}/hourglass_8__td_1__distill_td_1.yaml"

  # Untied Weights, No Distillation
#   "${CFG_ROOT}/hourglass_8__td_0_untied.yaml"
#   "${CFG_ROOT}/hourglass_8__td_0_25_untied.yaml"
#   "${CFG_ROOT}/hourglass_8__td_0_5_untied.yaml"
#   "${CFG_ROOT}/hourglass_8__td_0_9_untied.yaml"
#   "${CFG_ROOT}/hourglass_8__td_1_untied.yaml"
)


for cfg in "${cfg_list[@]}"
do
    for THRESHOLD_VAL in "${THRESHOLD_VALS[@]}"
    do
        cmd=( python pose_estimation/valid.py )   # create array with one element
        cmd+=( --cfg $cfg )
        cmd+=( --force_overwrite )
        cmd+=( --threshold $THRESHOLD_VAL )
        cmd+=( --dataset_key $DATASET_KEY )
        cmd+=( --load_best_ckpt )
    #     cmd+=( --vis_output_only )
        cmd+=( --save_all_data )
        cmd+=( --max_batch_logs $MAX_BATCH_LOGS )
    #     Run command
        "${cmd[@]}"
    done
done