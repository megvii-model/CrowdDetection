import numpy as np
import megengine.functional as F
import megengine.module as M

from config import config
from det_opr.utils import mask_to_inds
from det_opr.anchors_generator import AnchorGenerator
from det_opr.find_top_rpn_proposals import find_top_rpn_proposals
from det_opr.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_opr.loss_opr import softmax_loss, smooth_l1_loss

class RPN(M.Module):
    def __init__(self, rpn_channel=256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = M.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = M.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = M.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        fm_stride = 2 ** (len(features) + 1)
        for fm in features:
            layer_anchors = self.anchors_generator(fm, fm_stride)
            fm_stride = fm_stride // 2
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        rpn_rois, rpn_probs = find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info)

        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            #rpn_labels = rpn_labels.astype(np.int32)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)

            # rpn loss
            valid_masks = rpn_labels >= 0
            valid_inds = mask_to_inds(valid_masks)
            objectness_loss = softmax_loss(
                pred_cls_score.ai[valid_inds],
                rpn_labels.ai[valid_inds])
            #objectness_loss = objectness_loss * valid_masks

            pos_masks = rpn_labels > 0
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets,
                rpn_bbox_targets,
                config.rpn_smooth_l1_beta)
            localization_loss = localization_loss * pos_masks
            normalizer = 1.0 / (valid_masks.sum())
            loss_rpn_cls = objectness_loss.sum() * normalizer
            loss_rpn_loc = localization_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            return rpn_rois, loss_dict
        else:
            return rpn_rois

