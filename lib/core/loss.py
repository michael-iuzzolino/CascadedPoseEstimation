# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
  def __init__(self, use_target_weight):
    super(JointsMSELoss, self).__init__()
    self.criterion = nn.MSELoss(size_average=True)
    self.use_target_weight = use_target_weight

  def forward(self, output, target, target_weight):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, dim=1)
    heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, dim=1)
    loss = 0

    for idx in range(num_joints):
      heatmap_pred = heatmaps_pred[idx].squeeze()
      heatmap_gt = heatmaps_gt[idx].squeeze()
      if self.use_target_weight:
        loss += 0.5 * self.criterion(
            heatmap_pred.mul(target_weight[:, idx]),
            heatmap_gt.mul(target_weight[:, idx])
        )
      else:
        loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

    return loss / num_joints


class TDLambda_JointsMSELoss(nn.Module):
  def __init__(self, use_target_weight, n_timesteps, 
               lambda_val=1.0, tdl_mode="OSD", normalize_loss=True):
    super(TDLambda_JointsMSELoss, self).__init__()
    self.criterion = nn.MSELoss(size_average=True)
    self.use_target_weight = use_target_weight
    self.n_timesteps = n_timesteps
    self.lambda_val = lambda_val
    self.tdl_mode = tdl_mode
    self.normalize_loss = normalize_loss
    
    self.criterion = JointsMSELoss(use_target_weight)

  def forward(self, predictions, target, target_weight):
    loss = 0
    timestep_losses = torch.zeros(self.n_timesteps)
    
    for t in range(len(predictions)):
      pred_i = predictions[t]

      # First term
      sum_term = torch.zeros_like(pred_i)
      t_timesteps = list(range(t+1, self.n_timesteps))
      for i, n in enumerate(t_timesteps, 1):
        pred_k = predictions[n].detach().clone()
        softmax_i = F.softmax(pred_k, dim=1)
        sum_term = sum_term + self.lambda_val**(i - 1) * softmax_i

      # Final terms
      term_1 = (1 - self.lambda_val) * sum_term
      term_2 = self.lambda_val**(self.n_timesteps - t - 1) * target
      target_j = term_1 + term_2

      # Compute loss
      loss_i = self.criterion(pred_i, target_j, target_weight)
      
      # Aggregate loss
      if self.tdl_mode == "EWS":
        loss = loss + loss_i
      else:
        # Ignore first timestep loss (all 0's output)
        if t > 0:
          loss = loss + loss_i

      # Log loss item
      timestep_losses[t] = loss_i.item()
      
    # Normalize loss
    if self.normalize_loss:
      loss = loss / float(self.n_timesteps)
      
    return loss