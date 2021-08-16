# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from models import custom_ops
from models import layers as res_layers


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

      
class PoseResNet(nn.Module):
  """Resnet base class."""

  def __init__(self, block, layers, cfg, **kwargs):
    """Initialize resnet."""
    super(PoseResNet, self).__init__()
    self._layers_arch = layers
    
    extra = cfg.MODEL.EXTRA
    self.deconv_with_bias = extra.DECONV_WITH_BIAS

    self._cascaded = cfg.MODEL.CASCADED
    self._cascaded_scheme = extra.CASCADED_SCHEME # ", "parallel")
    self._time_bn = self._cascaded
    
    # TDL kwargs
    tdl_kwargs = {
        "tdl_mode": extra.TDL_MODE,
        "tdl_alpha": extra.TDL_ALPHA,
        "noise_var": extra.NOISE_VAR,
    }
    
    # Set up batch norm operation
    self._norm_layer_op = self._setup_bn_op()

    # Head layer
    self.res_layer_count = 0
    self.inplanes = 64
    self.layer0 = res_layers.HeadLayer(self.res_layer_count, 
                                       self.inplanes,
                                       self._norm_layer_op,
                                       cascaded=self._cascaded,
                                       time_bn=self._time_bn)
    self.res_layer_count += 1

    # Residual Layers
    self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **tdl_kwargs)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **tdl_kwargs)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **tdl_kwargs)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **tdl_kwargs)
    self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    
    # used for deconv layers
    self.deconv_layers = self._make_deconv_layer(
        extra.NUM_DECONV_LAYERS,
        extra.NUM_DECONV_FILTERS,
        extra.NUM_DECONV_KERNELS,
    )

    self.final_layer = nn.Conv2d(
        in_channels=extra.NUM_DECONV_FILTERS[-1],
        out_channels=cfg.MODEL.NUM_JOINTS,
        kernel_size=extra.FINAL_CONV_KERNEL,
        stride=1,
        padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
    )

    def init_weights(self, pretrained=''):
      if os.path.isfile(pretrained):
        logger.info('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
          if isinstance(m, nn.ConvTranspose2d):
            logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
            logger.info('=> init {}.bias as 0'.format(name))
            nn.init.normal_(m.weight, std=0.001)
            if self.deconv_with_bias:
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
            logger.info('=> init {}.weight as 1'.format(name))
            logger.info('=> init {}.bias as 0'.format(name))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        logger.info('=> init final conv weights from normal distribution')
        for m in self.final_layer.modules():
          if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
            logger.info('=> init {}.bias as 0'.format(name))
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

        # pretrained_state_dict = torch.load(pretrained)
        logger.info('=> loading pretrained model {}'.format(pretrained))
        # self.load_state_dict(pretrained_state_dict, strict=False)
        checkpoint = torch.load(pretrained)
        if isinstance(checkpoint, OrderedDict):
          state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
          state_dict_old = checkpoint['state_dict']
          state_dict = OrderedDict()
          # delete 'module.' because it is saved from DataParallel module
          for key in state_dict_old.keys():
            if key.startswith('module.'):
              # state_dict[key[7:]] = state_dict[key]
              # state_dict.pop(key)
              state_dict[key[7:]] = state_dict_old[key]
            else:
              state_dict[key] = state_dict_old[key]
        else:
          raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(pretrained))
        self.load_state_dict(state_dict, strict=False)
      else:
        logger.error('=> imagenet pretrained model dose not exist')
        logger.error('=> please download it first')
        raise ValueError('imagenet pretrained model does not exist')

  def _setup_bn_op(self):
    if self._cascaded and self._time_bn:
      self._norm_layer = custom_ops.BatchNorm2d

      # Setup batchnorm opts
      bn_opts = {
          "temporal_affine": False,
          "temporal_stats": True,
          "n_timesteps": self.timesteps
      }
      norm_layer_op = functools.partial(self._norm_layer, bn_opts)
    else:
      self._norm_layer = nn.BatchNorm2d
      norm_layer_op = self._norm_layer

    return norm_layer_op

  def _make_layer(self, block, planes, blocks, stride=1, **tdl_kwargs):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          custom_ops.conv1x1(self.inplanes, planes * block.expansion, stride),
      )
    layers = []
    layers.append(
        block(self.res_layer_count,
              self.inplanes,
              planes,
              stride,
              downsample,
              self._norm_layer_op,
              cascaded=self._cascaded,
              cascaded_scheme=self._cascaded_scheme,
              time_bn=self._time_bn,
              **tdl_kwargs))
    self.res_layer_count += 1
    
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(self.res_layer_count,
                self.inplanes,
                planes,
                norm_layer=self._norm_layer_op,
                cascaded=self._cascaded,
                cascaded_scheme=self._cascaded_scheme,
                time_bn=self._time_bn,
                **tdl_kwargs))
      self.res_layer_count += 1
    return nn.Sequential(*layers)

  def _get_deconv_cfg(self, deconv_kernel, index):
      if deconv_kernel == 4:
          padding = 1
          output_padding = 0
      elif deconv_kernel == 3:
          padding = 1
          output_padding = 1
      elif deconv_kernel == 2:
          padding = 0
          output_padding = 0

      return deconv_kernel, padding, output_padding

  def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
      assert num_layers == len(num_filters), \
          'ERROR: num_deconv_layers is different len(num_deconv_filters)'
      assert num_layers == len(num_kernels), \
          'ERROR: num_deconv_layers is different len(num_deconv_filters)'

      layers = []
      for i in range(num_layers):
          kernel, padding, output_padding = \
              self._get_deconv_cfg(num_kernels[i], i)

          planes = num_filters[i]
          layers.append(
              nn.ConvTranspose2d(
                  in_channels=self.inplanes,
                  out_channels=planes,
                  kernel_size=kernel,
                  stride=2,
                  padding=padding,
                  output_padding=output_padding,
                  bias=self.deconv_with_bias))
          layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
          layers.append(nn.ReLU(inplace=True))
          self.inplanes = planes

      return nn.Sequential(*layers)

  @property
  def timesteps(self):
    if self._cascaded:
      n_timesteps = np.sum(self._layers_arch) + 1
    else:
      n_timesteps = 1
    return n_timesteps

  def _set_time(self, t):
    self.layer0.set_time(t)
    for layer in self.layers:
      for block in layer:
        block.set_time(t)
  
  def set_target_inference_costs(self, normed_flops, target_inference_costs, 
                                 use_all=False):
    if use_all:
      print("Using all ICs!")
      selected_ICs = list(range(len(normed_flops)-1))
      IC_costs = normed_flops
    else:
      selected_ICs = []
      IC_costs = []
      for target_cost in target_inference_costs:
        diffs = []
        for normed_flop in normed_flops:
          abs_diff = np.abs(target_cost - normed_flop)
          diffs.append(abs_diff)
        min_idx = np.argmin(diffs)
        IC_cost = normed_flops[min_idx]
        IC_costs.append(IC_cost)
        selected_ICs.append(min_idx)
    self.selected_ICs = np.array(selected_ICs)
    self.IC_costs = np.array(IC_costs)

  def turn_off_IC(self):
    for k, params in self.named_parameters():
      if "IC" in k and "final" not in k:
        params.requires_grad = False
        
  def freeze_backbone(self, verbose=False):
    print("Freezing backbone param...")
    self.frozen_params = []
    self.unfrozen_params = []
    for k, params in self.named_parameters():
      if "IC" not in k:
        self.frozen_params.append(k)
        if verbose:
          print(f"\t{k} [frozen]")
        params.requires_grad = False
      else:
        self.unfrozen_params.append(k)
        
  def _forward(self, x, t=0):
    # Set time on all blocks
    if self._cascaded:
      self._set_time(t)

    # Head layer
    out = self.layer0(x)
    
    # Res Layers
    for layer in self.layers:
      out = layer(out)
      
    # Deconv
    out = self.deconv_layers(out)
    out = self.final_layer(out)

    return out
  
  def forward(self, x, t=0):
    return self._forward(x, t)
  

resnet_spec = {
    10: (res_layers.BasicBlock, [1, 1, 1, 1]),
    14: (res_layers.BasicBlock, [1, 2, 2, 1]),
    18: (res_layers.BasicBlock, [2, 2, 2, 2]),
    34: (res_layers.BasicBlock, [3, 4, 6, 3]),
    50: (res_layers.Bottleneck, [3, 4, 6, 3]),
    101: (res_layers.Bottleneck, [3, 4, 23, 3]),
    152: (res_layers.Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, cfg, **kwargs)
    
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
