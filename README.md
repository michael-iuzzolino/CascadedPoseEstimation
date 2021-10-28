# Train

Teacher TD

export CUDA_VISIBLE_DEVICES=0; python pose_estimation/train.py \
    --cfg experiments/mpii/hourglass/hourglass_4__td_0__double.yaml

export CUDA_VISIBLE_DEVICES=1; python pose_estimation/train.py \
    --cfg experiments/mpii/hourglass/hourglass_4__td_1__double.yaml

Student TD
export CUDA_VISIBLE_DEVICES=1; python pose_estimation/train.py \
    --cfg experiments/mpii/hourglass/hourglass_4__td_1.yaml

Distillation
export CUDA_VISIBLE_DEVICES=1; python pose_estimation/distillation_train.py \
    --cfg experiments/mpii/hourglass/hourglass_4__td_0__distill_td_0.yaml
    