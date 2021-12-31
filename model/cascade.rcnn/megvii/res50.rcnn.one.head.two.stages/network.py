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
from det_oprs.loss_opr import softmax_loss, softmax_loss_opr,smooth_l1_loss_rcnn_opr
import pdb
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):

        image = (image - torch.tensor(config.image_mean.reshape(1, -1, 1, 1)).type_as(image)) / (
                torch.tensor(config.image_std.reshape(1, -1, 1, 1)).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, rpn_loss_dict = self.RPN(fpn_fms, im_info, gt_boxes)
        # rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
        #         rpn_rois, im_info, gt_boxes, top_k=1)
        rcnn_loss_dict = self.RCNN(fpn_fms, rpn_rois, gt_boxes, im_info)

        loss_dict.update(rpn_loss_dict)
        loss_dict.update(rcnn_loss_dict)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class CascadeRCNN(nn.Module):

    def __init__(self, iou_thresh, nheads, stage):

        super().__init__()

        assert iou_thresh >= 0.5 and nheads > 0
        self.iou_thresh = iou_thresh
        self.nheads = nheads
        self.n = config.num_classes
        self.name = 'cascade_stage_{}'.format(stage)

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()

        self.n = config.num_classes
        self.p = nn.Linear(1024, 5 * self.n * nheads)
        self._init_weights()

    def _init_weights(self):

        for l in [self.fc1, self.fc2, self.p]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rois, gtboxes=None, im_info = None):

        rpn_fms = fpn_fms[1:]
        rpn_fms.reverse()
        rcnn_rois = rois
        if self.training:
            rcnn_rois, labels, bbox_targets = fpn_roi_target(rois, im_info, gtboxes, 
                self.iou_thresh, top_k=self.nheads)

        stride = [4, 8, 16, 32]
        pool5 = roi_pooler(rpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        pool5 = torch.flatten(pool5, start_dim=1)
        fc1 = self.relu(self.fc1(pool5))
        fc2 = self.relu(self.fc2(fc1))
        prob = self.p(fc2)

        loss = {}
        if self.training:
            # compute the loss function and then return 
            bbox_targets = bbox_targets.reshape(-1, 4) if self.nheads > 1 else bbox_targets
            labels = labels.view(-1)
            loss = self.compute_regular_loss(prob, bbox_targets, labels) if self.nheads < 2 else \
                self.compute_gemini_loss_opr(prob, bbox_targets, labels)
            pred_bboxes = self.recover_pred_boxes(rcnn_rois, prob, self.nheads)

            return loss, pred_bboxes
        
        else:
            # return the detection boxes and their scores
            pred_boxes = self.recover_pred_boxes(rcnn_rois, prob, self.nheads)
            return pred_boxes
    
    @torch.no_grad()
    def recover_pred_boxes(self, rcnn_rois, prob, nhead):

        n = prob.shape[0]
        prob = prob.reshape(n, nhead, -1)
        prob = prob.reshape(-1, prob.shape[2])
        cls_score, bbox_pred = prob[:, -self.n:], prob[:, :-self.n]
        cls_prob = torch.softmax(cls_score, dim=1)
        rois = rcnn_rois.unsqueeze(1).repeat(1, nhead, 1).reshape(-1, rcnn_rois.shape[1])
        bbox_pred = bbox_pred.reshape(n * nhead, -1, 4)
        
        pred_boxes = restore_bbox(rois[:, 1:5], bbox_pred, config = config)
        pred_boxes = torch.cat([pred_boxes, cls_prob.unsqueeze(2)], dim = 2)
        bid = rois[:, :1].unsqueeze(1).repeat(1, pred_boxes.shape[1], 1)
        pred_boxes = torch.cat([pred_boxes, bid], dim = 2)

        return pred_boxes.float()

    def compute_emd_loss_opr(self, a, b, bbox_targets, labels):
        
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

    def compute_gemini_loss_opr(self, prob, bbox_targets, labels):

        prob = prob.reshape(prob.shape[0], 2, -1)
        n, _, c = prob.shape
        prob = prob.permute(1, 0, 2)
        a, b = prob[0], prob[1]
        loss0 = self.compute_emd_loss_opr(a, b, bbox_targets, labels)
        loss1 = self.compute_emd_loss_opr(b, a, bbox_targets, labels)
        loss = torch.stack([loss0, loss1], dim = 1)
        emd_loss = loss.min(axis=1)[0].sum()/max(loss.shape[0], 1)
        loss = {'rcnn_emd_loss': emd_loss}
        return loss

    def compute_regular_loss(self, prob, bbox_targets, labels):

        offsets, cls_scores = prob[:,:-self.n], prob[:, -self.n:]
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        cls_loss = softmax_loss(cls_scores, labels)
        
        bbox_loss = smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, config.rcnn_smooth_l1_beta)

        bbox_loss = bbox_loss.sum() / torch.clamp((labels > 0).sum(), 1)
        loss = {}
        loss['{}_cls_loss'.format(self.name)] = cls_loss
        loss['{}_bbox_loss'.format(self.name)] = bbox_loss
        return loss

class RCNN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.n = config.num_classes

        self.iou_thresholds = [0.5, 0.5]
        self.nheads = [1, 1]
        self.heads = []
        for i, iou_thresh in enumerate(self.iou_thresholds):

            rcnn_head = CascadeRCNN(iou_thresh, self.nheads[i], i+1)
            self.heads.append(rcnn_head)
        self.heads = nn.ModuleList(self.heads)

    def _forward_train(self, fpn_fms, rcnn_rois, gtboxes, im_info):

        loss_dict = {}
        for i, _ in enumerate(self.iou_thresholds):

            loss, pred_boxes = self.heads[i](fpn_fms, rcnn_rois, gtboxes, im_info)
            rois = pred_boxes[:, 1]
            rcnn_rois = torch.cat([rois[:, 5:], rois[:,:4]], dim = 1).detach()
            loss_dict.update(loss)
        return loss_dict

    def _forward_test(self, fpn_fms, rcnn_rois, gtboxes=None, im_info=None):

        for i, _ in enumerate(self.iou_thresholds):

            pred_boxes = self.heads[i](fpn_fms, rcnn_rois)
            rois = pred_boxes[:, 1]
            rcnn_rois = torch.cat([rois[:, 5:], rois[:,:4]], dim = 1).detach()
        return pred_boxes[:, :, :5]

    def forward(self, fpn_fms, rcnn_rois, gtboxes = None, im_info=None):
        

        if self.training:
            
            return self._forward_train(fpn_fms, rcnn_rois, gtboxes, im_info)

        else:

            return self._forward_test(fpn_fms, rcnn_rois)
