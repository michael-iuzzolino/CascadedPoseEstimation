# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import sys

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.vis import log_outputs


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    n_timesteps = config.MODEL.EXTRA.N_HG_STACKS
    if config.MODEL.EXTRA.DOUBLE_STACK:
        n_timesteps *= 2
    accs = [AverageMeter() for _ in range(n_timesteps)]

    # switch to train mode
    model.train()
    end = time.time()

    for i, (x_data, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Set target weight
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # compute output
        outputs = model(x_data)

        # loss
        loss = criterion(outputs, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), x_data.size(0))

        for acc, output in zip(accs, outputs):
            _, avg_acc, cnt, pred = accuracy(
              output.detach().cpu().numpy(),
              target.detach().cpu().numpy()
            )
            acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.PRINT_FREQ == 0:
            msg = f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
            msg += f"Loss {losses.val:0.5f} ({losses.avg:0.5f})\t"
            msg += "Accuracy "
            for j, acc in enumerate(accs):
                msg += f"stack={j}: {acc.val:0.3f} ({acc.avg:0.3f}), "
            msg = msg[:-2]
            logger.info(msg)
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            
            save_debug_images(
              config, x_data, meta, target, pred * 4, output, prefix
            )


def validate(config, val_loader, val_dataset, model, criterion, output_dir):
    losses = AverageMeter()
    
    n_timesteps = config.MODEL.EXTRA.N_HG_STACKS
    if config.MODEL.EXTRA.DOUBLE_STACK:
        n_timesteps *= 2
    accs = [AverageMeter() for _ in range(n_timesteps)]

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    for i, (x_data, target, target_weight, meta) in enumerate(val_loader):
        num_images = x_data.size(0)

        center = meta['center'].numpy()
        scale = meta['scale'].numpy()
        score = meta['score'].numpy()

        # Set target and target_weight device
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # Compute outputs
        with torch.no_grad():
            outputs = model(x_data)

        # Compute loss
        loss = criterion(outputs, target, target_weight)
        losses.update(loss.item(), num_images)

        # Measure accuracy
        for acc, output in zip(accs, outputs):
            _, avg_acc, cnt, pred = accuracy(
                output.cpu().numpy(),
                target.cpu().numpy()
            )
            acc.update(avg_acc, cnt)
        
        
        output = outputs[-1]
        preds, maxvals = get_final_preds(
            config, 
            output.clone().cpu().numpy(), 
            center, 
            scale
        )

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals

        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(scale*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        if config.DATASET.DATASET == 'posetrack':
            filenames.extend(meta['filename'])
            imgnums.extend(meta['imgnum'].numpy())

        idx += num_images

        if i % config.PRINT_FREQ == 0:
            msg = f"Test: [{i}/{len(val_loader)}]\t"
            msg += f"Loss {losses.val:0.5f} ({losses.avg:0.5f})\t"
            msg += "Accuracy "
            for i, acc in enumerate(accs):
                msg += f"stack={i}: {acc.val:0.3f} ({acc.avg:0.3f}), "
            msg = msg[:-2]
            logger.info(msg)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
            save_debug_images(
              config, x_data, meta, target, pred*4, output, prefix
            )

    name_values, perf_indicator = val_dataset.evaluate(
        config, 
        all_preds, 
        output_dir, 
        all_boxes, 
        image_path,
        filenames, 
        imgnums
    )

    _, full_arch_name = get_model_name(config)
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, full_arch_name)
    else:
        _print_name_value(name_values, full_arch_name)

    return perf_indicator


def test(
    config, 
    val_loader, 
    model, 
    threshold=0.5, 
    save_root="", 
    max_batch_logs=5, 
    save_all_data=False
):
    # switch to evaluate mode
    model.eval()

    if save_all_data:
        full_data_root = os.path.join(save_root, "full_data")
        if not os.path.exists(full_data_root):
            os.makedirs(full_data_root)
        
    n_timesteps = config.MODEL.EXTRA.N_HG_STACKS
    if config.MODEL.EXTRA.DOUBLE_STACK:
        n_timesteps *= 2
    accs = [AverageMeter() for _ in range(n_timesteps)]
    log_flops = True
    with torch.no_grad():
        for i, (x_data, target, target_weight, meta) in enumerate(val_loader):
            sys.stdout.write(f"\rBatch {i+1:,}/{len(val_loader):,}, Accuracy: {accs[-1].avg}...")
            sys.stdout.flush()

            # Set target and target_weight device
            target = target.cuda(non_blocking=True)
        
            # Compute outputs
            with torch.no_grad():
                outputs = model(x_data, log_flops=log_flops)
            total_flops = model.module.get_flops()

            if save_all_data:
                save_path = os.path.join(save_root, f"batch_{i:04d}.pt")
                torch.save({
                    "outputs": outputs,
                }, save_path)
            
            # Measure accuracy
            for acc, output in zip(accs, outputs):
                _, avg_acc, cnt, pred = accuracy(
                    output.cpu().numpy(),
                    target.cpu().numpy(),
                    threshold=threshold,
                )
                acc.update(avg_acc, cnt)

            # Log data
            if save_root and i < max_batch_logs:
                log_outputs(x_data, target, outputs, meta, save_root)
            log_flops = False
    print("\n")
    for i, acc in enumerate(accs):
        print(f"Stack={i}: {acc.avg * 100:0.3f}%")

    result = [acc.avg * 100 for acc in accs]

    return result, total_flops


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
