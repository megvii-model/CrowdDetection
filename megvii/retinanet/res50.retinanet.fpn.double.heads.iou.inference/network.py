import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.retina_anchor_target import retina_anchor_target
from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr, bbox_transform_opr
from det_oprs.utils import get_padded_tensor
from module.generate_anchors import generate_anchors
from rpn_anchor_target_opr import rpn_anchor_target_opr
from det_oprs.loss_opr import softmax_loss, iou_l1_loss, smooth_l1_loss_retina,  \
    sigmoid_cross_entropy_retina
import pdb

class RetinaNet_AnchorV2(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def generate_anchors_opr(self, fm_3x3, fm_stride,
        anchor_scales=(8, 16, 32, 64, 128), 
        anchor_ratios=(1, 2, 3), base_size = 4):

        np_anchors = generate_anchors(
            base_size=base_size,
            ratios=np.array(anchor_ratios),
            scales=np.array(anchor_scales))
        
        device = fm_3x3.device
        anchors = torch.tensor(np_anchors).to(device)
        height, width = fm_3x3.shape[2:]
        shift_x = torch.linspace(0, width - 1, width).to(device) * fm_stride
        shift_y = torch.linspace(0, height - 1, height).to(device) * fm_stride

        broad_shift_x = shift_x.view(1, -1).repeat(height, 1).view(-1)
        broad_shift_y = shift_y.view(-1, 1).repeat(1, width).view(-1)
        shifts = torch.stack([broad_shift_x, broad_shift_y, broad_shift_x, broad_shift_y], dim=1)

        all_anchors = anchors.unsqueeze(0) + shifts.unsqueeze(1)
        all_anchors = all_anchors.view(-1, anchors.shape[1])
        return all_anchors

    def forward(self, fpn_fms):

        all_anchors_list = []
        fm_stride = [8, 16, 32, 64, 128]
        fm_stride.reverse()

        for i, fm_3x3 in enumerate(fpn_fms):
            
            anchor_scales = np.array(config.anchor_base_scale) * fm_stride[i]
            all_anchors = self.generate_anchors_opr(fm_3x3, fm_stride[i], anchor_scales,
                config.anchor_aspect_ratios, base_size = 4)
            all_anchors_list.append(all_anchors)
        return all_anchors_list

class Network(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.R_Head = RetinaNet_Head()
        self.R_Anchor = RetinaNet_AnchorV2()
        self.R_Criteria = RetinaNetCriteriaV2()
    
    def pre_processing(self, image):

        mean_tensor = torch.tensor(config.image_mean).type_as(image).view(1, 3, 1, 1)
        std_tensor = torch.tensor(config.image_std).type_as(image).view(1, 3, 1, 1)
        image = (image - mean_tensor) / std_tensor
        image = get_padded_tensor(image, 64)
        
        return image

    def forward(self, image, im_info, gt_boxes=None):
        
        # pre-processing the data
        image = self.pre_processing(image)
        # do inference
        # stride: 128,64,32,16,8, p7->p3
        fpn_fms = self.FPN(image)
        anchors_list = self.R_Anchor(fpn_fms)
        pred_cls_list, rpn_num_prob_list, pred_reg_list, rpn_iou_list = self.R_Head(fpn_fms)
        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(
                pred_cls_list, rpn_num_prob_list, pred_reg_list, anchors_list,
                rpn_iou_list, gt_boxes, im_info)
            return loss_dict
        else:

            results = self.forward_inference(anchors_list, pred_cls_list,
                pred_reg_list, rpn_iou_list, im_info)
            return results
    
    def forward_inference(self, anchors_list, pred_cls_list,
        pred_reg_list, rpn_iou_list, im_info):

        shapes = [r.shape for r in anchors_list]
        all_anchors = torch.cat(anchors_list, dim = 0)
        cls_scores = torch.cat(pred_cls_list, dim = 1)
        bbox_pred = torch.cat(pred_reg_list, dim = 1)
        rpn_iou_prob = torch.cat(rpn_iou_list, dim = 1)
        

        anchors = all_anchors.unsqueeze(1).repeat(1, 2, 1).view(-1, all_anchors.shape[1])
        n = cls_scores.shape[0]
        cls_scores = cls_scores.reshape(n, -1, config.num_classes-1)
        rpn_iou_prob = rpn_iou_prob.reshape(n, -1, config.num_classes-1)
        bbox_pred = bbox_pred.reshape(n, -1, 4)

        res = []
        for i in range(n):

            pred_bbox = bbox_transform_inv_opr(anchors, bbox_pred[i])
            cls_prob =  cls_scores[i]
            rpn_iou = rpn_iou_prob[i]
            dtboxes = torch.cat([pred_bbox, cls_prob, rpn_iou], dim=1)
            res.append(dtboxes.unsqueeze(0))

        results = torch.cat(res, dim = 0)
        return results

class RetinaNetCriteriaV2(nn.Module):

    def __init__(self):
        
        super().__init__()

    @torch.no_grad()
    def anchor_iou_target_opr(self, boxes, im_info, all_anchors,
            rpn_bbox_offsets):


        n = rpn_bbox_offsets.shape[0]

        res = []
        for i in range(n):

            gtboxes = boxes[i, :im_info[i, 5].long()]
            offsets =  rpn_bbox_offsets[i].reshape(-1, 4)
            m = offsets.shape[0]
            anchors = all_anchors.unsqueeze(1).repeat(1, 2, 1).view(-1, all_anchors.shape[1])

            dtboxes = bbox_transform_inv_opr(anchors[:,:4], offsets[:, :4])
            overlaps = box_overlap_opr(dtboxes, gtboxes[:, :4])
            ignore_mask = 1 - gtboxes[:, 4].eq(config.anchor_ignore_label).float().unsqueeze(0)
            overlaps = overlaps * ignore_mask
            overlaps = overlaps.view(-1, 2, overlaps.shape[1]).permute(1, 0, 2)
            
            a, b = overlaps[0], overlaps[1]
            index = torch.argmax(a, dim = 1)
            a = torch.gather(a, 1, index.unsqueeze(1))
            b = torch.scatter(b, 1, index.unsqueeze(1), 0)
            index = torch.argmax(b, dim = 1)
            b = torch.gather(b, 1, index.unsqueeze(1))
            value = torch.cat([a, b], dim = 1)
            res.append(value.unsqueeze(0))

        result = torch.cat(res, 0)
        return result

    def forward(self, pred_cls_list, rpn_num_prob_list, pred_reg_list,
        anchors_list, rpn_iou_list, boxes, im_info):

        all_anchors_list = [torch.cat([a, i*torch.ones(a.shape[0], 1).to(a.device)], dim=1) 
            for i, a in enumerate(anchors_list)]
        all_anchors_final = torch.cat(all_anchors_list, dim = 0)
        
        rpn_bbox_offset_final = torch.cat(pred_reg_list, dim = 1)
        rpn_cls_prob_final = torch.cat(pred_cls_list, dim = 1)
        rpn_iou_prob_final = torch.cat(rpn_iou_list, dim = 1)
        rpn_num_per_points_final = torch.cat(rpn_num_prob_list, dim = 1)

        rpn_labels, rpn_target_boxes = rpn_anchor_target_opr(boxes, im_info, all_anchors_final)
        ious_target = self.anchor_iou_target_opr(boxes, im_info, all_anchors_final,
            rpn_bbox_offset_final)

        n = rpn_labels.shape[0]
        target_boxes =  rpn_target_boxes.view(n, -1, 4)
        rpn_cls_prob_final = rpn_cls_prob_final.view(n, -1, 1)
        offsets_final = rpn_bbox_offset_final.view(n, -1, 4)
        
        rpn_labels = rpn_labels.permute(2, 0, 1)
        a, b = rpn_labels[0], rpn_labels[1]

        ignores = b - a.eq(0).float() * b.eq(0).eq(0)
        labels = torch.stack([a, ignores], dim = 2).view(n, -1)
        cls_loss = sigmoid_cross_entropy_retina(rpn_cls_prob_final, 
                labels, alpha = 0.25, gamma = 2)
        rpn_bbox_loss = smooth_l1_loss_retina(offsets_final, target_boxes, labels)

        rpn_labels = labels.view(n, -1, 2)
        rpn_iou_loss = iou_l1_loss(rpn_iou_prob_final, ious_target, rpn_labels)

        # whether one anchor produce one proposal or two.
        nlabels = ((labels.reshape(n, -1, 2) > 0).sum(2)).flatten() - 1
        c = rpn_num_per_points_final.shape[2]
        num_per_anchor = rpn_num_per_points_final.reshape(-1, c)

        rpn_num_per_points_final = rpn_num_per_points_final.reshape(-1, c)
        nlabels = nlabels.view(-1)
        rpn_num_loss = softmax_loss(rpn_num_per_points_final, nlabels)       
        
        loss_dict = {}
        loss_dict['rpn_cls_loss'] = cls_loss
        loss_dict['rpn_bbox_loss'] = 2 * rpn_bbox_loss
        loss_dict['rpn_iou_loss'] = 2 * rpn_iou_loss
        loss_dict['rpn_num_loss'] = rpn_num_loss
        return loss_dict


class RetinaNet_Head(nn.Module):
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        # predictor
        self.cls_score = nn.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1) * 2,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 4 * 2,
            kernel_size=3, stride=1, padding=1)

        self.iou_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 2,
            kernel_size = 3, stride=1, padding = 1)

        self.num_pred = nn.Conv2d(in_channels,
            config.num_cell_anchors * 2,
            kernel_size = 3, stride=1, padding = 1)
        self._init_weights()

    def _init_weights(self):
        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.num_pred,
                self.cls_score, self.bbox_pred, self.iou_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
   
        cls_prob_list, rpn_num_prob_list, pred_bbox_list, rpn_iou_prob_list = [], [], [], []
        for feature in features:

            rpn_cls_conv = self.cls_subnet(feature)
            cls_score = self.cls_score(rpn_cls_conv)
            rpn_num_prob = self.num_pred(rpn_cls_conv)

            cls_prob = torch.sigmoid(cls_score)

            rpn_box_conv = self.bbox_subnet(feature)
            offsets = self.bbox_pred(rpn_box_conv)
            rpn_iou_prob = self.iou_pred(rpn_box_conv)

            cls_prob_list.append(cls_prob)
            pred_bbox_list.append(offsets)
            rpn_iou_prob_list.append(rpn_iou_prob)
            rpn_num_prob_list.append(rpn_num_prob)


        assert cls_prob_list[0].ndim == 4
        pred_cls_list = [
            _.permute(0, 2, 3, 1).reshape(_.shape[0], -1, 2*(config.num_classes-1))
            for _ in cls_prob_list]
        pred_reg_list = [
            _.permute(0, 2, 3, 1).reshape(_.shape[0], -1, 4*2)
            for _ in pred_bbox_list]
        rpn_iou_list = [
            _.permute(0, 2, 3, 1).reshape(_.shape[0], -1, 2*(config.num_classes-1))
            for _ in rpn_iou_prob_list]

        rpn_num_prob_list = [
            _.permute(0, 2, 3, 1).reshape(_.shape[0], -1, 2*(config.num_classes-1))
            for _ in rpn_num_prob_list]
        return pred_cls_list, rpn_num_prob_list, pred_reg_list, rpn_iou_list
