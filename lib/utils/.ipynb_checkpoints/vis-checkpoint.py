# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import cv2

from core.inference import get_max_preds


def combine_heatmap_joints(X, resize=()):
    X_joined = X.sum(axis=0)
    if resize:
        X_joined = cv2.resize(X_joined, resize, interpolation=cv2.INTER_CUBIC)
    return X_joined

def log_outputs(x_data, target, outputs, meta, save_root):
    renorm = lambda ele: (ele - ele.min()) / (ele.max() - ele.min())
    
    for i in range(x_data.shape[0]):
        save_path = os.path.join(save_root, f"batch_{i:04d}.pt")
        Xi = x_data[i].permute(1,2,0)
        Xi = renorm(Xi).cpu().detach().numpy()
        Xi = cv2.cvtColor(Xi, cv2.COLOR_BGR2RGB)

        yi = target[i].cpu().detach().numpy()
        y_target = combine_heatmap_joints(yi, resize=Xi.shape[:2])
        y_target = renorm(y_target)
        outputs_i = outputs[:,i].cpu().detach().numpy()

        out_combos = []
        pred_joint_imgs = []
        combo_joint_imgs = []
        for out_i in outputs_i:
            out_combo = combine_heatmap_joints(out_i, resize=Xi.shape[:2])
            out_combo = torch.tensor(out_combo).unsqueeze(dim=0)
            out_combo = renorm(out_combo)
            out_combos.append(out_combo)
            out_i = out_i[np.newaxis, ...]
            joint_pred, _ = get_max_preds(out_i)
            joint_img_i = save_batch_image_with_joints(
                batch_image=torch.tensor(Xi).permute(2,0,1).unsqueeze(dim=0), 
                batch_joints=joint_pred * 4, 
                batch_joints_vis=meta["joints_vis"],
                joint_color=(255,0,0),
            )
            combo_img_i = save_batch_image_with_joints(
                batch_image=torch.tensor(Xi).permute(2,0,1).unsqueeze(dim=0), 
                batch_joints=joint_pred * 4, 
                batch_joints_2=meta["joints"], 
                batch_joints_vis=meta["joints_vis"],
                joint_color=(255,0,0),
                joint_color_2=(0,255,0),
            )
            pred_joint_imgs.append(joint_img_i)
            combo_joint_imgs.append(combo_img_i)

        torch.save({
            "x_data": Xi,
            "target": y_target,
            "predictions": out_combos,
            "pred_joint_imgs": pred_joint_imgs,
            "combo_joint_imgs": combo_joint_imgs,
        }, save_path)
    
def save_batch_image_with_joints(
    batch_image, 
    batch_joints, 
    batch_joints_vis,
    batch_joints_2=[],
    file_name="", 
    nrow=8, 
    padding=2,
    joint_color=(255, 0, 0),
    joint_color_2=(0,255,0),
):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            if len(batch_joints_2):
                joints_2 = batch_joints_2[k]
            else:
                joints_2 = joints
            joints_vis = batch_joints_vis[k]

            for joint, joint_2, joint_vis in zip(joints, joints_2, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, joint_color, 2)
                if len(batch_joints_2):
                    joint_2[0] = x * width + padding + joint_2[0]
                    joint_2[1] = y * height + padding + joint_2[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint_2[0]), int(joint_2[1])), 2, joint_color_2, 2)
            k = k + 1
    if file_name:
        cv2.imwrite(file_name, ndarr)
    return ndarr


def save_batch_heatmaps(
    batch_image,
    batch_heatmaps, 
    file_name,
    normalize=True
):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(
    config, 
    input, 
    meta, 
    target, 
    joints_pred, 
    output,
    prefix
):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
