import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.module as M

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.roi_pool import roi_pool
from det_opr.bbox_opr import bbox_transform_inv_opr
from det_opr.fpn_roi_target import fpn_roi_target
from det_opr.cascade_roi_target import cascade_roi_target
from det_opr.loss_opr import softmax_loss, smooth_l1_loss
from det_opr.utils import get_padded_tensor

class Network(M.Module):
    def __init__(self):
        super().__init__()
        # ----------------------- build the backbone ------------------------ #
        self.resnet50 = ResNet50()
        # ------------ freeze the weights of resnet stage1 and stage 2 ------ #
        if config.backbone_freeze_at >= 1:
            for p in self.resnet50.conv1.parameters():
                p.requires_grad = False
        if config.backbone_freeze_at >= 2:
            for p in self.resnet50.layer1.parameters():
                p.requires_grad = False
        # -------------------------- build the FPN -------------------------- #
        self.backbone = FPN(self.resnet50)
        # -------------------------- build the RPN -------------------------- #
        self.RPN = RPN(config.rpn_channel)
        # ----------------------- build the RCNN head ----------------------- #
        self.Cascade_0 = Cascade('cascade_0')
        #self.Cascade_1 = Cascade('cascade_1')
        self.RCNN = RCNN()
        # -------------------------- input Tensor --------------------------- #
        self.inputs = {
            "image": mge.tensor(
                np.random.random([2, 3, 224, 224]).astype(np.float32), dtype="float32",
            ),
            "im_info": mge.tensor(
                np.random.random([2, 5]).astype(np.float32), dtype="float32",
            ),
            "gt_boxes": mge.tensor(
                np.random.random([2, 100, 5]).astype(np.float32), dtype="float32",
            ),
        }

    def forward(self, inputs):
        images = inputs['image']
        im_info = inputs['im_info']
        gt_boxes = inputs['gt_boxes']
        # process the images
        normed_images = (
            images - config.image_mean[None, :, None, None]
        ) / config.image_std[None, :, None, None]
        normed_images = get_padded_tensor(normed_images, 64)
        if self.training:
            return self._forward_train(normed_images, im_info, gt_boxes)
        else:
            return self._forward_test(normed_images, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        # stride: 64,32,16,8,4, p6->p2
        fpn_fms = self.backbone(image)
        rpn_rois, loss_dict_rpn = \
            self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
            rpn_rois, im_info, gt_boxes, top_k=1)
        proposals, loss_dict_cascade_0 = self.Cascade_0(
            fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)
        cascade_rois, cascade_labels, cascade_bbox_targets = cascade_roi_target(
            proposals, im_info, gt_boxes, pos_threshold=0.6, top_k=2)
        #proposals, loss_dict_cascade_1 = self.Cascade_1(
        #    fpn_fms, cascade_rois, cascade_labels, cascade_bbox_targets)
        #cascade_rois, cascade_labels, cascade_bbox_targets = cascade_roi_target(
        #    proposals, im_info, gt_boxes, pos_threshold=0.7, top_k=2)
        loss_dict_rcnn = self.RCNN(
            fpn_fms, cascade_rois, cascade_labels, cascade_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_cascade_0)
        #loss_dict.update(loss_dict_cascade_1)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.backbone(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        proposals, cascade_cls_0 = self.Cascade_0(fpn_fms, rpn_rois)
        pred_bbox = self.RCNN(fpn_fms, proposals)
        pred_score = pred_bbox[:, -1][:, None]
        pred_bbox = pred_bbox[:, :-1] / im_info[0, 2]
        pred_bbox = F.concat((pred_bbox, pred_score), axis=1)
        return pred_bbox

class Cascade(M.Module):
    def __init__(self, name):
        super().__init__()
        self.stage_name = name
        # roi head
        self.fc1 = M.Linear(256*7*7, 1024)
        self.fc2 = M.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            M.init.msra_uniform_(l.weight, a=1)
            M.init.fill_(l.bias, 0)
        # box predictor
        self.pred_cls = M.Linear(1024, 2)
        self.pred_delta = M.Linear(1024, 4)
        for l in [self.pred_cls]:
            M.init.normal_(l.weight, std=0.01)
            M.init.normal_(l.bias, 0)
        for l in [self.pred_delta]:
            M.init.normal_(l.weight, std=0.001)
            M.init.normal_(l.bias, 0)

    def forward(self, fpn_fms, proposals, labels=None, bbox_targets=None):
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        #pool_features = roi_pooler(fpn_fms, proposals, stride, (7, 7), "ROIAlignV2")
        pool_features, proposals, labels, bbox_targets = roi_pool(
                fpn_fms, proposals, stride, (7, 7), 'roi_align',
                labels, bbox_targets)
        flatten_feature = F.flatten(pool_features, start_axis=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))
        pred_cls = self.pred_cls(roi_feature)
        pred_delta = self.pred_delta(roi_feature)
        if self.training:
            # loss for regression
            labels = labels.astype(np.int32).reshape(-1)
            # mulitple class to one
            pos_masks = labels > 0
            localization_loss = smooth_l1_loss(
                pred_delta,
                bbox_targets,
                config.rcnn_smooth_l1_beta)
            localization_loss = localization_loss * pos_masks
            # loss for classification
            valid_masks = labels >= 0
            objectness_loss = softmax_loss(
                pred_cls,
                labels)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / (valid_masks.sum())
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_dict = {}
            loss_dict[self.stage_name + '_cls'] = loss_rcnn_cls
            loss_dict[self.stage_name + '_loc'] = loss_rcnn_loc
            pred_bbox = restore_bbox(proposals[:, 1:5], pred_delta, True)
            pred_proposals = F.zero_grad(F.concat([proposals[:, 0].reshape(-1, 1), pred_bbox], axis=1))
            return pred_proposals, loss_dict
        else:
            pred_scores = F.softmax(pred_cls)[:, 1].reshape(-1, 1)
            pred_bbox = restore_bbox(proposals[:, 1:5], pred_delta, True)
            pred_proposals = F.concat([proposals[:, 0].reshape(-1, 1), pred_bbox], axis=1)
            return pred_proposals, pred_scores

class RCNN(M.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = M.Linear(256*7*7, 1024)
        self.fc2 = M.Linear(1024, 1024)
        self.fc3 = M.Linear(1044, 1024)
        for l in [self.fc1, self.fc2, self.fc3]:
            M.init.msra_uniform_(l.weight, a=1)
            M.init.fill_(l.bias, 0)
        # box predictor
        self.emd_pred_cls_0 = M.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = M.Linear(1024, (config.num_classes - 1) * 4)
        self.emd_pred_cls_1 = M.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = M.Linear(1024, (config.num_classes - 1) * 4)
        self.ref_pred_cls_0 = M.Linear(1024, config.num_classes)
        self.ref_pred_delta_0 = M.Linear(1024, (config.num_classes - 1) * 4)
        self.ref_pred_cls_1 = M.Linear(1024, config.num_classes)
        self.ref_pred_delta_1 = M.Linear(1024, (config.num_classes - 1) * 4)
        for l in [self.emd_pred_cls_0, self.emd_pred_cls_1, 
                self.ref_pred_cls_0, self.ref_pred_cls_1]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)
        for l in [self.emd_pred_delta_0, self.emd_pred_delta_1,
                self.ref_pred_delta_0, self.ref_pred_delta_1]:
            M.init.normal_(l.weight, std=0.001)
            M.init.fill_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features, rcnn_rois, labels, bbox_targets = roi_pool(
                fpn_fms, rcnn_rois, stride, (7, 7), 'roi_align',
                labels, bbox_targets)
        flatten_feature = F.flatten(pool_features, start_axis=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))
        pred_emd_pred_cls_0 = self.emd_pred_cls_0(roi_feature)
        pred_emd_pred_delta_0 = self.emd_pred_delta_0(roi_feature)
        pred_emd_pred_cls_1 = self.emd_pred_cls_1(roi_feature)
        pred_emd_pred_delta_1 = self.emd_pred_delta_1(roi_feature)
        pred_emd_scores_0 = F.softmax(pred_emd_pred_cls_0)
        pred_emd_scores_1 = F.softmax(pred_emd_pred_cls_1)
        # make refine feature
        box_0 = F.concat((pred_emd_pred_delta_0,
            pred_emd_scores_0[:, 1][:, None]), axis=1)[:, None, :]
        box_1 = F.concat((pred_emd_pred_delta_1,
            pred_emd_scores_1[:, 1][:, None]), axis=1)[:, None, :]
        boxes_feature_0 = box_0.broadcast(
                box_0.shapeof()[0], 4, box_0.shapeof()[-1]).reshape(box_0.shapeof()[0], -1)
        boxes_feature_1 = box_1.broadcast(
                box_1.shapeof()[0], 4, box_1.shapeof()[-1]).reshape(box_1.shapeof()[0], -1)
        boxes_feature_0 = F.concat((roi_feature, boxes_feature_0), axis=1)
        boxes_feature_1 = F.concat((roi_feature, boxes_feature_1), axis=1)
        refine_feature_0 = F.relu(self.fc3(boxes_feature_0))
        refine_feature_1 = F.relu(self.fc3(boxes_feature_1))
        # refine
        pred_ref_pred_cls_0 = self.ref_pred_cls_0(refine_feature_0)
        pred_ref_pred_delta_0 = self.ref_pred_delta_0(refine_feature_0)
        pred_ref_pred_cls_1 = self.ref_pred_cls_1(refine_feature_1)
        pred_ref_pred_delta_1 = self.ref_pred_delta_1(refine_feature_1)
        if self.training:
            loss0 = emd_loss(
                        pred_emd_pred_delta_0, pred_emd_pred_cls_0,
                        pred_emd_pred_delta_1, pred_emd_pred_cls_1,
                        bbox_targets, labels)
            loss1 = emd_loss(
                        pred_emd_pred_delta_1, pred_emd_pred_cls_1,
                        pred_emd_pred_delta_0, pred_emd_pred_cls_0,
                        bbox_targets, labels)
            loss2 = emd_loss(
                        pred_ref_pred_delta_0, pred_ref_pred_cls_0,
                        pred_ref_pred_delta_1, pred_ref_pred_cls_1,
                        bbox_targets, labels)
            loss3 = emd_loss(
                        pred_ref_pred_delta_1, pred_ref_pred_cls_1,
                        pred_ref_pred_delta_0, pred_ref_pred_cls_0,
                        bbox_targets, labels)
            loss_rcnn = F.concat([loss0, loss1], axis=1)
            loss_ref = F.concat([loss2, loss3], axis=1)
            indices_rcnn = F.argmin(loss_rcnn, axis=1)
            indices_ref = F.argmin(loss_ref, axis=1)
            loss_rcnn = F.indexing_one_hot(loss_rcnn, indices_rcnn, 1)
            loss_ref = F.indexing_one_hot(loss_ref, indices_ref, 1)
            loss_rcnn = loss_rcnn.sum()/loss_rcnn.shapeof()[0]
            loss_ref = loss_ref.sum()/loss_ref.shapeof()[0]
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_rcnn
            loss_dict['loss_ref_emd'] = loss_ref
            return loss_dict
        else:
            pred_ref_scores_0 = F.softmax(pred_ref_pred_cls_0)
            pred_ref_scores_1 = F.softmax(pred_ref_pred_cls_1)
            pred_bbox_0 = restore_bbox(rcnn_rois[:, 1:5], pred_ref_pred_delta_0, True)
            pred_bbox_1 = restore_bbox(rcnn_rois[:, 1:5], pred_ref_pred_delta_1, True)
            pred_bbox_0 = F.concat([pred_bbox_0, pred_ref_scores_0[:, 1].reshape(-1,1)], axis=1)
            pred_bbox_1 = F.concat([pred_bbox_1, pred_ref_scores_1[:, 1].reshape(-1,1)], axis=1)
            pred_bbox = F.concat((pred_bbox_0, pred_bbox_1), axis=1).reshape(-1,5)
            return pred_bbox

def emd_loss(p_b0, p_c0, p_b1, p_c1, targets, labels):
    pred_box = F.concat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shapeof()[-1])
    pred_score = F.concat([p_c0, p_c1], axis=1).reshape(-1, p_c0.shapeof()[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.reshape(-1)
    fg_masks = F.greater(labels, 0)
    non_ignore_masks = F.greater_equal(labels, 0)
    # loss for regression
    loss_box_reg = smooth_l1_loss(
        pred_box,
        targets,
        config.rcnn_smooth_l1_beta)
    # loss for classification
    loss_cls = softmax_loss(pred_score, labels)
    loss = loss_cls*non_ignore_masks + loss_box_reg * fg_masks
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = mge.tensor(config.bbox_normalize_stds[None, :])
        mean_opr = mge.tensor(config.bbox_normalize_means[None, :])
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

