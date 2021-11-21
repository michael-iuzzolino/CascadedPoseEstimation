# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate, test
from utils.utils import create_experiment_directory

import dataset
import models.pose_stacked_hg
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--max_batch_logs',
                        help='Max # of batches to save data from',
                        default=5,
                        type=int)
    parser.add_argument('--dataset_key',
                        help='dataset key: valid, test',
                        default="valid",
                        type=str)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--result_root',
                        default="/hdd/mliuzzolino/TDPoseEstimation/results/",
                        help='Root for results',
                        type=str)
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='Accuracy threshold [default=0.5]')
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--load_best_ckpt',
                        help='Load best checkpoint [default: load final]',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--force_overwrite',
                        help='Force overwrite',
                        action='store_true')
    parser.add_argument('--vis_output_only',
                        help='Visualize output only; dont save results',
                        action='store_true')
    parser.add_argument('--save_all_data',
                        help='Save all data',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def get_state_dict(output_dir, config, use_best=False):
  if config.TEST.MODEL_FILE:
    state_dict = torch.load(config.TEST.MODEL_FILE)
  else:
    ckpt_path = os.path.join(output_dir, f"final_state.pth.tar")
    
    if os.path.exists(ckpt_path) and not use_best:
      state_dict = torch.load(ckpt_path)
    else:
      ckpt_path = os.path.join(output_dir, f"model_best.pth.tar")
      state_dict = torch.load(ckpt_path)
  
  if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
        
  return state_dict


def main():
    # Setup args and config
    args = parse_args()
    reset_config(config, args)
    
    basename = os.path.basename(args.cfg)

    output_dir = create_experiment_directory(
        config, 
        args.cfg, 
        distillation="distill" in basename,
        make_dir=False,
    )
    if ("distill" in basename) and ("untied" in basename):
      pass
    elif "untied" in basename:
      output_dir = f"{output_dir}__no_skip"
    elif "distill" in basename:
      pass
    else:
      if "distill" in basename:
        teacher_td = basename.split("__distill")[1].split(".")[0].split("td_")[1]
        if "_" in teacher_td:
          teacher_td = teacher_td.replace("_", ".")
        teacher_td = float(teacher_td)
        output_dir = f"{output_dir}__distill__TD_{teacher_td}"
    
    # Setup output dir
    output_dir_tmp = os.path.sep.join(output_dir.split(os.path.sep)[1:])
    final_result_root = os.path.join(args.result_root, output_dir_tmp)
    if not os.path.exists(final_result_root):
      os.makedirs(final_result_root)
    
    # Save output save root
    save_root = final_result_root.replace("/hdd/", "/hdd3/")
#     save_root = os.path.join(save_root, args.dataset_key)
    
    if not os.path.exists(save_root):
      os.makedirs(save_root)

    # Set results save path
    save_path = os.path.join(final_result_root, f"result__{args.threshold}.npy")
    if os.path.exists(save_path) and not args.force_overwrite:
      print(f"Already exists @ {save_path}. Exiting!")
      exit()
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    # Setup model
    model = models.pose_stacked_hg.get_pose_net(config)
      
    # Load state dict
    state_dict = get_state_dict(
      output_dir, 
      config, 
      use_best=args.load_best_ckpt
    )
    
    # Load previous model
    model.load_state_dict(state_dict)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Data loading code
    normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225],
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    result, total_flops = test(
        config,
        valid_loader,
        model,
        threshold=args.threshold,
        save_root=save_root,
        max_batch_logs=args.max_batch_logs,
        save_all_data=args.save_all_data,
    )

    if result is not None:
      flops_save_path = os.path.join(final_result_root, "flops.pt")
      print(f"Saving flops to {flops_save_path}")
      torch.save(total_flops, flops_save_path)
      
      if not args.vis_output_only:
        print(f"Saving to {save_path}")
        np.save(save_path, result)
        print("Complete.")


if __name__ == '__main__':
    main()
