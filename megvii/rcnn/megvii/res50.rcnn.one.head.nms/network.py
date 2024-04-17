import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr, restore_bbox
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.utils import get_padded_tensor
from det_oprs.loss_opr import softmax_loss_opr, softmax_loss, smooth_l1_loss_rcnn_opr
import pdb

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                rpn_rois, im_info, gt_boxes, top_k=1)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        
        # box predictor
        n = 5 * config.num_classes
        self.p = nn.Linear(1024, n)
        self.n = config.num_classes

    def _init_weights(self):

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        for l in [self.p]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        
        # input p2-p5
        fpn_fms = fpn_fms[1:]
        fpn_fms.reverse()
        
        stride = [4, 8, 16, 32]
        pool5 = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        pool5 = torch.flatten(pool5, start_dim=1)
        
        fc1 = self.relu(self.fc1(pool5))
        fc2 = self.relu(self.fc2(fc1))
        prob = self.p(fc2)

        if self.training:
            # loss for regression
            offsets, cls_score = prob[:, :-self.n], prob[:,-self.n:]

            n = offsets.shape[0]
            bbox_targets, labels = bbox_targets.reshape(-1, 4), labels.flatten()
            cls_loss = softmax_loss_opr(cls_score, labels)
            
            offsets = offsets.reshape(n, -1, 4)
            reg_loss = smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
                labels, sigma = config.rcnn_smooth_l1_beta)

            cls_num = (labels > -1).sum()
            reg_num = (labels > 0).sum()
            reg_loss  = reg_loss.sum() / torch.clamp(reg_num, 1)
            cls_loss = cls_loss.sum() / torch.clamp(cls_num, 1)
            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = reg_loss
            loss_dict['loss_rcnn_cls'] = cls_loss
            return loss_dict
        
        else:

            cls_score, bbox_pred = prob[:, -self.n:], prob[:, :-self.n]
            cls_prob = torch.softmax(cls_score, dim=1)
            rois = rcnn_rois[:,1:5]
            bbox_pred = bbox_pred.reshape(-1, self.n, 4)
            pred_bbox = restore_bbox(rois, bbox_pred, config = config)

            cls_prob = cls_prob.unsqueeze(2)
            pred_bbox = torch.cat([pred_bbox, cls_prob], dim=2)

            return pred_bbox

    def compute_det_loss_opr(self, a, b, bbox_targets, labels):
        
        labels = labels.long().flatten()
        c = a.shape[1]
        prob = torch.stack([a, b], dim=1).reshape(-1, c)
        offsets, cls_score = prob[:, :-self.n], prob[:,-self.n:]
        cls_loss = softmax_loss_opr(cls_score, labels)
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        reg_loss = smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, sigma = config.rcnn_smooth_l1_beta)

        vlabel = 1 - ((labels < 0).view(-1, 2).sum(axis=1) > 1).float()
        loss = (cls_loss + 2 * reg_loss).view(-1, 2).sum(dim=1) * vlabel
        return loss

    def emd_loss_opr(self, prob, bbox_targets, labels):
        
        n, c = prob.shape
        prob = prob.reshape(-1, 2, c).permute(1, 0, 2)
        a, b = prob[0], prob[1]
        loss0 = self.compute_det_loss_opr(a, b, bbox_targets, labels)
        loss1 = self.compute_det_loss_opr(b, a, bbox_targets, labels)
        loss = torch.stack([loss0, loss1], dim = 1)
        emd_loss = loss.min(axis=1)[0].sum()/max(loss.shape[0], 1)
        return emd_loss
    

    def restore_bbox(self, rois, deltas, unnormalize=True):

        assert rois.shape[0] == deltas.shape[0]
        n, m, c = deltas.shape
        rois = rois.unsqueeze(1).repeat(1, m, 1).reshape(-1, rois.shape[1])
        deltas = deltas.reshape(-1, c)
        if unnormalize:
            std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
            mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
            deltas = deltas * std_opr
            deltas = deltas + mean_opr
        pred_bbox = bbox_transform_inv_opr(rois, deltas)
        pred_bbox = pred_bbox.reshape(-1, m, c)
        return pred_bbox
