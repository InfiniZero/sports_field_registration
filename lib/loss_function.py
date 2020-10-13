import torch
import numpy as np
import torch.nn.functional as F
import cv2


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def init_h_loss_func(pred_h, gt_h):
    loss = torch.pow(pred_h - gt_h, 2)
    loss = torch.sum(loss)
    return loss


def refine_h_loss_func(pred_deltah, gt_deltah, pred_iou, gt_iou):
    loss_deltah = init_h_loss_func(pred_deltah, gt_deltah)
    loss_iou = init_h_loss_func(pred_iou, gt_iou)
    loss = loss_deltah + loss_iou
    return loss, loss_deltah, loss_iou


def seg_loss_func(pred_mask, gt_mask, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    pred = torch.sigmoid(pred_mask)
    dice = dice_loss(pred, gt_mask)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss
