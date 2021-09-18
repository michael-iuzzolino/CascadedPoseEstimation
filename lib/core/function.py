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


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  accs = [AverageMeter() for _ in range(config.MODEL.EXTRA.N_HG_STACKS)]

  # switch to train mode
  model.train()
  end = time.time()

  for i, (input, target, target_weight, meta) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    # Set target weight
    target = target.cuda(non_blocking=True)
    target_weight = target_weight.cuda(non_blocking=True)

    if config.MODEL.CASCADED:
      outputs = []
      for t in range(config.MODEL.N_TIMESTEPS):
        output = model(input, t)
        outputs.append(output)

      loss = criterion(outputs, target, target_weight)

    else:
      # compute output
      outputs = model(input)

      # loss
      loss = criterion(outputs, target, target_weight)

    # compute gradient and do update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure accuracy and record loss
    losses.update(loss.item(), input.size(0))

    for acc, output in zip(accs, outputs):
      _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                       target.detach().cpu().numpy())
      acc.update(avg_acc, cnt)
      
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    
    if i % config.PRINT_FREQ == 0:
      msg = f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
#       msg += f"Time {batch_time.val:0.3f}s ({batch_time.avg:0.3f}s)\t"
#       msg += f"Speed {input.size(0)/batch_time.val:0.1f} samples/s\t"
#       msg += f"Data: {data_time.val:0.3f}s ({data_time.avg:0.3f}s)\t"
      msg += f"Loss {losses.val:0.5f} ({losses.avg:0.5f})\t"
      msg += "Accuracy "
      for i, acc in enumerate(accs):
        msg += f"stack={i}: {acc.val:0.3f} ({acc.avg:0.3f}), "
      msg = msg[:-2]
      logger.info(msg)

      writer = writer_dict['writer']
      global_steps = writer_dict['train_global_steps']
      writer.add_scalar('train_loss', losses.val, global_steps)
      for i, acc in enumerate(accs):
        writer.add_scalar(f'train_acc_{i}', acc.val, global_steps)
      writer_dict['train_global_steps'] = global_steps + 1

      prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
      save_debug_images(config, input, meta, target, pred*4, output,
                        prefix)

      
def revert_flip(output_flipped, flip_pairs):
    output_flipped = flip_back(output_flipped.cpu().numpy(),flip_pairs)
    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

    # feature is not aligned, shift flipped heatmap for higher accuracy
    if config.TEST.SHIFT_HEATMAP:
      output_flipped[:, :, :, 1:] = \
        output_flipped.clone()[:, :, :, 0:-1]

    output = (output + output_flipped) * 0.5
    return output
  

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
  batch_time = AverageMeter()
  losses = AverageMeter()
  accs = [AverageMeter() for _ in range(config.MODEL.EXTRA.N_HG_STACKS)]

  # switch to evaluate mode
  model.eval()

  num_samples = len(val_dataset)
  all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
  all_boxes = np.zeros((num_samples, 6))
  image_path = []
  filenames = []
  imgnums = []
  idx = 0
  
  for i, (input, target, target_weight, meta) in enumerate(val_loader):
    num_images = input.size(0)

    center = meta['center'].numpy()
    scale = meta['scale'].numpy()
    score = meta['score'].numpy()

    # Set target and target_weight device
    target = target.cuda(non_blocking=True)
    target_weight = target_weight.cuda(non_blocking=True)

    # Compute outputs
    with torch.no_grad():
      outputs = model(input)

    # Compute loss
    loss = criterion(outputs, target, target_weight)
    losses.update(loss.item(), num_images)

    # Measure accuracy
    for acc, output in zip(accs, outputs):
      _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                       target.cpu().numpy())
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
      save_debug_images(config, input, meta, target, pred*4, output,
                        prefix)

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

  if writer_dict:
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', losses.avg, global_steps)
    for i, acc in enumerate(accs):
      writer.add_scalar(f'valid_acc_{i}', acc.val, global_steps)
    if isinstance(name_values, list):
      for name_value in name_values:
        writer.add_scalars('valid', dict(name_value), global_steps)
    else:
      writer.add_scalars('valid', dict(name_values), global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

  return perf_indicator


def input_flip_fxn(input):
  # this part is ugly, because pytorch has not supported negative index
  # input_flipped = model(input[:, :, :, ::-1])
  input_flipped = np.flip(input.cpu().numpy(), 3).copy()
  input_flipped = torch.from_numpy(input_flipped).cuda()
  return input_flipped


def test(config, val_loader, val_dataset, model, threshold=0.5):
  # switch to evaluate mode
  model.eval()
  
  if config.MODEL.CASCADED:
    acc_metric = [AverageMeter() for _ in range(config.MODEL.N_TIMESTEPS)]
  else:
    acc_metric = AverageMeter()
  
  with torch.no_grad():
    for i, (input, target, _, _) in enumerate(val_loader):
      sys.stdout.write(f"\rBatch {i+1:,}/{len(val_loader):,}...")
      sys.stdout.flush()

      # Set target and target_weight device
      target = target.cuda(non_blocking=True)
    
      input_fxn = lambda x: x
      output_fxn = lambda x: x

      inputs = input_fxn(input)

      output = model(inputs)
      output = output_fxn(output)

      # Measure accuracy
      _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                       target.cpu().numpy(),
                                       threshold=threshold)
      acc_metric.update(avg_acc, cnt)
    
  result = acc_metric.avg * 100
  print(f"Average: {result:0.2f}%")
  
  return result


def generate_outputs(config, val_loader, val_dataset, model):
  # switch to evaluate mode
  model.eval()
  all_outputs = []
  all_targets = []
  with torch.no_grad():
    end = time.time()
    for i, (input, target, _, _) in enumerate(val_loader):
      sys.stdout.write(f"\rBatch {i+1:,}/{len(val_loader):,}...")
      sys.stdout.flush()
      
      # Set target
      target = target.cuda(non_blocking=True)
      
      if config.TEST.FLIP_TEST:
        input_fxn = lambda x: input_flip_fxn(x)
        output_fxn = lambda x: revert_flip(x, val_dataset.flip_pairs)
      else:
        input_fxn = lambda x: x
        output_fxn = lambda x: x

      inputs = input_fxn(input)

      if config.MODEL.CASCADED:
        outputs = []
        for t in range(config.MODEL.N_TIMESTEPS):
          output = model(inputs, t)
          output = output_fxn(output)
          outputs.append(output)
      else:
        output = model(inputs)
        output = output_fxn(output)
      
      all_outputs.append(output)
      all_targets.append(target)

  return all_outputs, all_targets
  

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
