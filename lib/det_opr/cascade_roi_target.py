# -*- coding: utf-8 -*-
import megengine as mge
import megengine.random as rand
import megengine.functional as F

import numpy as np
from config import config
from det_opr.utils import mask_to_inds
from det_opr.bbox_opr import box_overlap_opr, bbox_transform_opr, box_overlap_ignore_opr


def cascade_roi_target(rpn_rois, im_info, gt_boxes, pos_threshold=0.5, top_k=1):
    return_rois = []
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :im_info[bid, 5], :]
        batch_inds = mge.ones((gt_boxes_perimg.shapeof()[0], 1)) * bid
        #if config.proposal_append_gt:
        gt_rois = F.concat([batch_inds, gt_boxes_perimg[:, :4]], axis=1)
        batch_roi_mask = rpn_rois[:, 0] == bid
        batch_roi_inds = mask_to_inds(batch_roi_mask)
        all_rois = F.concat([rpn_rois.ai[batch_roi_inds], gt_rois], axis=0)
        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois[:, 1:5], gt_boxes_perimg)
        overlaps_normal, overlaps_normal_indices = F.argsort(overlaps_normal, descending=True)
        overlaps_ignore, overlaps_ignore_indices = F.argsort(overlaps_ignore, descending=True)
        # gt max and indices, ignore max and indices
        max_overlaps_normal = overlaps_normal[:, :top_k].reshape(-1)
        gt_assignment_normal = overlaps_normal_indices[:, :top_k].reshape(-1)
        max_overlaps_ignore = overlaps_ignore[:, :top_k].reshape(-1)
        gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].reshape(-1)
        # cons masks
        ignore_assign_mask = (max_overlaps_normal < config.fg_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
        max_overlaps = max_overlaps_normal * (1 - ignore_assign_mask) + \
                max_overlaps_ignore * ignore_assign_mask
        gt_assignment = gt_assignment_normal * (1- ignore_assign_mask) + \
                gt_assignment_ignore * ignore_assign_mask
        gt_assignment = gt_assignment.astype(np.int32)
        labels = gt_boxes_perimg.ai[gt_assignment, 4]
        fg_mask = (max_overlaps >= config.fg_threshold) * (1 - F.equal(labels, config.ignore_label))
        bg_mask = (max_overlaps < config.bg_threshold_high) * (
                max_overlaps >= config.bg_threshold_low)
        fg_mask = fg_mask.reshape(-1, top_k)
        bg_mask = bg_mask.reshape(-1, top_k)
        #pos_max = config.num_rois * config.fg_ratio
        #fg_inds_mask = _bernoulli_sample_masks(fg_mask[:, 0], pos_max, 1)
        #neg_max = config.num_rois - fg_inds_mask.sum()
        #bg_inds_mask = _bernoulli_sample_masks(bg_mask[:, 0], neg_max, 1)
        labels = labels * fg_mask.reshape(-1)
        #keep_mask = fg_inds_mask + bg_inds_mask
        #keep_inds = mask_to_inds(keep_mask)
        #keep_inds = keep_inds[:F.minimum(config.num_rois, keep_inds.shapeof()[0])]
        # labels
        labels = labels.reshape(-1, top_k)
        gt_assignment = gt_assignment.reshape(-1, top_k).reshape(-1)
        target_boxes = gt_boxes_perimg.ai[gt_assignment, :4]
        #rois = all_rois.ai[keep_inds]
        target_shape = (all_rois.shapeof()[0], top_k, all_rois.shapeof()[-1])
        target_rois = F.add_axis(all_rois, 1).broadcast(target_shape).reshape(-1, all_rois.shapeof()[-1])
        bbox_targets = bbox_transform_opr(target_rois[:, 1:5], target_boxes)
        if config.rcnn_bbox_normalize_targets:
            std_opr = mge.tensor(config.bbox_normalize_stds[None, :])
            mean_opr = mge.tensor(config.bbox_normalize_means[None, :])
            minus_opr = mean_opr / std_opr
            bbox_targets = bbox_targets / std_opr - minus_opr
        bbox_targets = bbox_targets.reshape(-1, top_k * 4)
        return_rois.append(all_rois)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
    if config.batch_per_gpu == 1:
        return F.zero_grad(all_rois), F.zero_grad(labels), F.zero_grad(bbox_targets)
    else:
        return_rois = F.concat(return_rois, axis=0)
        return_labels = F.concat(return_labels, axis=0)
        return_bbox_targets = F.concat(return_bbox_targets, axis=0)
        return F.zero_grad(return_rois), F.zero_grad(return_labels), F.zero_grad(return_bbox_targets)