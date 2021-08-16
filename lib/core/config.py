# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# pose_resnet related params
POSE_RESNET = edict()
POSE_RESNET.CASCADED = False
POSE_RESNET.CASCADED_SCHEME = "parallel"  # parallel, serial
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2

# pose_resnet related params
CASCADED_UNET = edict()
# CASCADED_UNET.NUM_LAYERS = 50
# CASCADED_UNET.DECONV_WITH_BIAS = False
# CASCADED_UNET.NUM_DECONV_LAYERS = 3
# CASCADED_UNET.NUM_DECONV_FILTERS = [256, 256, 256]
# CASCADED_UNET.NUM_DECONV_KERNELS = [4, 4, 4]
# CASCADED_UNET.FINAL_CONV_KERNEL = 1
# CASCADED_UNET.TARGET_TYPE = 'gaussian'
# CASCADED_UNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
# CASCADED_UNET.SIGMA = 2

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
    'cascaded_pose_resnet': POSE_RESNET,
    'cascaded_unet': POSE_RESNET,
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.CASCADED = False
config.MODEL.NAME = 'cascaded_unet'  # 'pose_resnet', 'cascaded_unet'
config.MODEL.INIT_WEIGHTS = False
config.MODEL.PRETRAINED = ''
config.MODEL.NUM_JOINTS = 16
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.TD_LAMBDA = 1.0
config.LOSS.NORMALIZE = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True

config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    td_lambda = cfg.LOSS.TD_LAMBDA
    
    if 'pose_resnet' in name:
        if cfg.MODEL.CASCADED:
            name = f'{name}_{extra.NUM_LAYERS}_cascaded_td({td_lambda})'
            deconv_suffix = ''.join(f'd{num_filters}' for num_filters in extra.NUM_DECONV_FILTERS)
            full_name = f'{cfg.MODEL.IMAGE_SIZE[1]}x{cfg.MODEL.IMAGE_SIZE[0]}_{name}_{deconv_suffix}'
            full_name += f"_cascaded_td({td_lambda})"
        else:
            name = f'{name}_{extra.NUM_LAYERS}'
            deconv_suffix = ''.join(f'd{num_filters}' for num_filters in extra.NUM_DECONV_FILTERS)
            full_name = f'{cfg.MODEL.IMAGE_SIZE[1]}x{cfg.MODEL.IMAGE_SIZE[0]}_{name}_{deconv_suffix}'
    elif name == "cascaded_unet":
        if cfg.MODEL.CASCADED:
            name = f"Cascaded UNet_x{extra.N_HG_STACKS}_td({td_lambda})"
            full_name = "Cascaded UNet [full_name]"
        else:
            name = "UNet"
            full_name = f"UNet_x{extra.N_HG_STACKS} [full_name]"
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
