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
  def __init__(self, use_target_weight=False):
    super(JointsMSELoss, self).__init__()
    self.criterion = nn.MSELoss(size_average=True)
    self.use_target_weight = use_target_weight

  def forward(self, output, target, target_weight=None):
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
  def __init__(self, use_target_weight, lambda_val=1.0, normalize_loss=False):
    super(TDLambda_JointsMSELoss, self).__init__()
    self.criterion = nn.MSELoss(size_average=True)
    self.use_target_weight = use_target_weight
    self.lambda_val = lambda_val
    self.normalize_loss = normalize_loss
    
    self.criterion = JointsMSELoss(use_target_weight)

  def forward(self, predictions, target, target_weight=None):
    loss = 0
    n_timesteps = len(predictions)
    timestep_losses = torch.zeros(n_timesteps)
    
    for t in range(len(predictions)):
      pred_i = predictions[t]

      # First term
      sum_term = torch.zeros_like(pred_i)
      t_timesteps = list(range(t+1, n_timesteps))
      for i, n in enumerate(t_timesteps, 1):
        pred_k = predictions[n].detach().clone()
        softmax_i = F.softmax(pred_k, dim=1)
        sum_term = sum_term + self.lambda_val**(i - 1) * softmax_i

      # Final terms
      term_1 = (1 - self.lambda_val) * sum_term
      term_2 = self.lambda_val**(n_timesteps - t - 1) * target
      target_j = term_1 + term_2

      # Compute loss
      loss_i = self.criterion(pred_i, target_j, target_weight)
      
      # Aggregate loss
      loss = loss + loss_i

      # Log loss item
      timestep_losses[t] = loss_i.item()
      
    # Normalize loss
#     if self.normalize_loss:
#       loss = loss / float(n_timesteps)
      
    return loss


class JointsMSELoss_Distillation(nn.Module):
  def __init__(
      self, 
      use_target_weight, 
      lambda_val=1.0, 
      normalize_loss=False, 
      alpha=0.5
    ):
    super(JointsMSELoss_Distillation, self).__init__()
    self.alpha = alpha
    self._mse_loss = TDLambda_JointsMSELoss(
        use_target_weight=use_target_weight,
        lambda_val=lambda_val,
        normalize_loss=normalize_loss,
    )
    self._teacher_criterion = TDLambda_JointsMSELoss(
        use_target_weight=False,
        lambda_val=1.0,
        normalize_loss=False,
    )

  def forward(self, outputs, teacher_target, target, target_weight):
    teacher_loss = self._teacher_criterion(outputs, teacher_target)
    mse_loss = self._mse_loss(outputs, target, target_weight)
    loss = self.alpha * teacher_loss + (1 - self.alpha) * mse_loss
    return loss