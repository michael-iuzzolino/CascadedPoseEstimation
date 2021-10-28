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
import shutil

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
from core.config import get_model_name
from core.loss import JointsMSELoss_Distillation
from core.distillation_function import train, validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_experiment_directory

import dataset
import models.pose_stacked_hg


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
    parser.add_argument('--alpha',
                        help='alpha for distillation loss',
                        default=0.5,
                        type=float)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
       config.WORKERS = args.workers

    
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


def setup_teacher(config, args, gpus):
    original_cfg = args.cfg
    args.cfg = "experiments/mpii/hourglass/hourglass_4__td_1__double.yaml"
  #   args.cfg = "experiments/mpii/hourglass/hourglass_8__td_1.yaml"
    update_config(args.cfg)
    teacher_model = models.pose_stacked_hg.get_pose_net(config, is_train=False)
    
    output_dir = create_experiment_directory(
        config, 
        args.cfg, 
        distillation=False,
        make_dir=False,
    )
    
    # Load state dict
    state_dict = get_state_dict(output_dir, config, use_best=True)

    # Load previous model
    teacher_model.load_state_dict(state_dict)

    # Set cfg back to original
    args.cfg = original_cfg
    update_config(args.cfg)
    
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpus).cuda()
    
    return teacher_model

  
def main():
    args = parse_args()
    reset_config(config, args)

    print("Setting up output experimental directory")
    output_dir = create_experiment_directory(
        config, 
        args.cfg, 
        distillation=True,
    )

    print("Initializing model...")
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # Setup model
    model = models.pose_stacked_hg.get_pose_net(config, is_train=True)
    
    # copy model file
    print("Copying model file...")
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        output_dir
    )

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                            3,
                            config.MODEL.IMAGE_SIZE[1],
                            config.MODEL.IMAGE_SIZE[0]))

    # Setup parallel model
    print("Parallelizing model...")
    gpus = [int(i) for i in config.GPUS.split(',')]
    print(f"GPUS: {gpus}")
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    
    # Setup teacher
    teacher_model = setup_teacher(config, args, gpus)

    print("Setting up criterion, optimizer, and LR scheduling...")
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss_Distillation(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT, 
        lambda_val=config.LOSS.TD_LAMBDA, 
        normalize_loss=config.LOSS.NORMALIZE,
        alpha=config.LOSS.DISTILLATION_ALPHA,
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    print("Setting up datasets...")
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
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

    print("Setting up dataset loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    print("Training model...")
    best_perf = 0.0
    best_model = False
    for epoch_i in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        # train for one epoch
        train(
            config, 
            train_loader, 
            model, 
            teacher_model,
            criterion, 
            optimizer, 
            epoch_i,
            output_dir,
        )

        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(
            config, 
            valid_loader, 
            valid_dataset, 
            model,
            teacher_model,
            criterion, 
            output_dir,
        )

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        save_dict = {
            'epoch_i': epoch_i + 1,
            'model': get_model_name(config),
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(save_dict, best_model, output_dir)

    # Final ckpt save
    save_dict = {
        'epoch_i': epoch_i + 1,
        'model': get_model_name(config),
        'state_dict': model.module.state_dict(),
        'perf': perf_indicator,
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(
        save_dict, 
        False, 
        output_dir, 
        filename='final_state.pth.tar'
    )
    

if __name__ == '__main__':
    main()
