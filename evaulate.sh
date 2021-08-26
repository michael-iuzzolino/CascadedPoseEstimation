#!/bin/bash
THRESHOLDS=(0.0 0.1 0.2 0.25 0.3 0.4 0.5 0.75 1.0) 

CFG="/home/michael/CascadedPoseEstimation/experiments/mpii/resnet18/cascaded__td_1_parallel.yaml"
# CFG="/home/michael/CascadedPoseEstimation/experiments/mpii/resnet18/baseline.yaml"
RESULTS_ROOT="/hdd/mliuzzolino/CascadedPoseEstimation/q_results/"
for THRESHOLD in "${THRESHOLDS[@]}"
do
    cmd=( python pose_estimation/valid.py )   # create array with one element
    cmd+=( --cfg $CFG )
    cmd+=( --threshold $THRESHOLD )
    cmd+=( --result_root $RESULTS_ROOT )

    # Run command
    "${cmd[@]}"
done