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
  acc = AverageMeter()

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
        output = model(input, t, is_train=True)
        outputs.append(output)

      loss = criterion(outputs, target, target_weight)

    else:
      # compute output
      output = model(input)

      # loss
      loss = criterion(output, target, target_weight)

    # compute gradient and do update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure accuracy and record loss
    losses.update(loss.item(), input.size(0))

    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                     target.detach().cpu().numpy())
    acc.update(avg_acc, cnt)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    
    if i % config.PRINT_FREQ == 0:
      msg = 'Epoch: [{0}][{1}/{2}]\t' \
            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            'Speed {speed:.1f} samples/s\t' \
            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0)/batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
      logger.info(msg)

      writer = writer_dict['writer']
      global_steps = writer_dict['train_global_steps']
      writer.add_scalar('train_loss', losses.val, global_steps)
      writer.add_scalar('train_acc', acc.val, global_steps)
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
  acc = AverageMeter()

  # switch to evaluate mode
  model.eval()

  num_samples = len(val_dataset)
  all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                       dtype=np.float32)
  all_boxes = np.zeros((num_samples, 6))
  image_path = []
  filenames = []
  imgnums = []
  idx = 0
  with torch.no_grad():
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
      num_images = input.size(0)
      
      # Set target and target_weight device
      target = target.cuda(non_blocking=True)
      target_weight = target_weight.cuda(non_blocking=True)
      
      # Compute outputs
      if config.TEST.FLIP_TEST:
        # this part is ugly, because pytorch has not supported negative index
        # input_flipped = model(input[:, :, :, ::-1])
        input_flipped = np.flip(input.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda()

        if config.MODEL.CASCADED:
          ouputs = []
          for t in range(config.MODEL.N_TIMESTEPS):
            output_flipped = model(input_flipped, t)
            output = revert_flip(output_flipped, val_dataset.flip_pairs)
            outputs.append(output)
        else:
          output_flipped = model(input_flipped)
          output = revert_flip(output_flipped, val_dataset.flip_pairs)
        
      else:
        if config.MODEL.CASCADED:
          outputs = []
          for t in range(config.MODEL.N_TIMESTEPS):
            output = model(input, t)
            outputs.append(output)
        else:
          output = model(input)
      
      # Compute loss
      if config.MODEL.CASCADED:
        loss = criterion(outputs, target, target_weight)
      else:
        loss = criterion(output, target, target_weight)
      losses.update(loss.item(), num_images)
  
      # Measure accuracy
      _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                       target.cpu().numpy())

      acc.update(avg_acc, cnt)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      c = meta['center'].numpy()
      s = meta['scale'].numpy()
      score = meta['score'].numpy()

      preds, maxvals = get_final_preds(
          config, 
          output.clone().cpu().numpy(), 
          c, 
          s
      )

      all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
      all_preds[idx:idx + num_images, :, 2:3] = maxvals
      
      # double check this all_boxes parts
      all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
      all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
      all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
      all_boxes[idx:idx + num_images, 5] = score
      image_path.extend(meta['image'])
      
      if config.DATASET.DATASET == 'posetrack':
        filenames.extend(meta['filename'])
        imgnums.extend(meta['imgnum'].numpy())

      idx += num_images

      if i % config.PRINT_FREQ == 0:
        msg = 'Test: [{0}/{1}]\t' \
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time,
                  loss=losses, acc=acc)
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
      writer.add_scalar('valid_acc', acc.avg, global_steps)
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

      # Compute loss
      if config.MODEL.CASCADED:
        for acc_met, output in zip(acc_metric, outputs):
          # Measure accuracy
          _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                           target.cpu().numpy(),
                                           threshold=threshold)
          acc_met.update(avg_acc, cnt)
      else:
        # Measure accuracy
        _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                         target.cpu().numpy(),
                                         threshold=threshold)
        acc_metric.update(avg_acc, cnt)
    
  if config.MODEL.CASCADED:
    print("Performance")
    for t, acc_met in enumerate(acc_metric):
      avg_val = acc_met.avg * 100
      print(f"{t}: {avg_val:0.2f}")
    result = [ele.avg * 100 for ele in acc_metric]
  else:
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
