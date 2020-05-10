import megengine as mge
import megengine.random as rand
import megengine.functional as F
import numpy as np

from det_opr.bbox_opr import box_overlap_opr, bbox_transform_opr
from det_opr.utils import mask_to_inds
from config import config

def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):
    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []
    for bid in range(config.batch_per_gpu):
        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []
        for i in range(len(pred_cls_score_list)):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid] \
                .dimshuffle(1, 2, 0).reshape(-1, 2)
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid] \
                .dimshuffle(1, 2, 0).reshape(-1, 4)
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)
        batch_pred_cls_score = F.concat(batch_pred_cls_score_list, axis=0)
        batch_pred_bbox_offsets = F.concat(batch_pred_bbox_offsets_list, axis=0)
        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)
    final_pred_cls_score = F.concat(final_pred_cls_score_list, axis=0)
    final_pred_bbox_offsets = F.concat(final_pred_bbox_offsets_list, axis=0)
    return final_pred_cls_score, final_pred_bbox_offsets

def fpn_anchor_target_opr_core_impl(
        gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    ignore_label = config.ignore_label
    # get the gt boxes
    valid_gt_boxes = gt_boxes[:im_info[5], :]
    non_ignore_mask = valid_gt_boxes[:, -1] > 0
    non_ignore_inds = mask_to_inds(non_ignore_mask) 
    valid_gt_boxes = valid_gt_boxes.ai[non_ignore_inds]
    # compute the iou matrix
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])
    # match the dtboxes
    a_shp0 = anchors.shape[0]
    max_overlaps = F.max(overlaps, axis=1)
    argmax_overlaps = F.argmax(overlaps, axis=1)
    # all ignore
    labels = mge.ones(a_shp0).astype(np.int32) * ignore_label
    # set negative ones
    labels = labels * (max_overlaps >= config.rpn_negative_overlap)
    # set positive ones
    fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    const_one = mge.tensor(1.0)
    if allow_low_quality_matches:
        # match the max gt
        gt_max_overlaps = F.max(overlaps, axis=0)
        gt_argmax_overlaps = F.argmax(overlaps, axis=0)
        g_shp0 = valid_gt_boxes.shapeof()[0]
        gt_id = F.linspace(0, g_shp0 - 1, g_shp0).astype(np.int32)
        argmax_overlaps = argmax_overlaps.set_ai(gt_id)[gt_argmax_overlaps]
        max_overlaps = max_overlaps.set_ai(const_one.broadcast(g_shp0))[gt_argmax_overlaps]
        fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    # set positive ones
    fg_mask_ind = mask_to_inds(fg_mask)
    labels = labels.set_ai(const_one.broadcast(fg_mask_ind.shapeof()))[fg_mask_ind]
    # compute the targets
    bbox_targets = bbox_transform_opr(
        anchors, valid_gt_boxes.ai[argmax_overlaps, :4])
    if config.rpn_bbox_normalize_targets:
        std_opr = mge.tensor(config.bbox_normalize_stds[None, :])
        mean_opr = mge.tensor(config.bbox_normalize_means[None, :])
        minus_opr = mean_opr / std_opr
        bbox_targets = bbox_targets / std_opr - minus_opr
    return labels, bbox_targets

def fpn_anchor_target(boxes, im_info, all_anchors_list):
    final_labels_list = []
    final_bbox_targets_list = []
    for bid in range(config.batch_per_gpu):
        batch_labels_list = []
        batch_bbox_targets_list = []
        for i in range(len(all_anchors_list)):
            anchors_perlvl = all_anchors_list[i]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)
            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)
        # here we samples the rpn_labels
        concated_batch_labels = F.concat(batch_labels_list, axis=0)
        concated_batch_bbox_targets = F.concat(batch_bbox_targets_list, axis=0)
        # sample labels
        num_positive = config.num_sample_anchors * config.positive_anchor_ratio
        concated_batch_labels = _bernoulli_sample_labels(concated_batch_labels,
                num_positive, 1, config.ignore_label)
        num_positive = F.equal(concated_batch_labels, 1).sum()
        num_negative = config.num_sample_anchors - num_positive
        concated_batch_labels = _bernoulli_sample_labels(concated_batch_labels,
                num_negative, 0, config.ignore_label)

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)
    final_labels = F.concat(final_labels_list, axis=0)
    final_bbox_targets = F.concat(final_bbox_targets_list, axis=0)
    return F.zero_grad(final_labels), F.zero_grad(final_bbox_targets)

def _bernoulli_sample_labels(
        labels, num_samples, sample_value, ignore_label=-1):
    """ Using the bernoulli sampling method"""
    sample_label_mask = F.equal(labels, sample_value)
    num_mask = sample_label_mask.sum()
    num_final_samples = F.minimum(num_mask, num_samples)
    # here, we use the bernoulli probability to sample the anchors
    sample_prob = num_final_samples / num_mask
    uniform_rng = rand.uniform(sample_label_mask.shapeof()[0])
    disable_mask = (uniform_rng >= sample_prob) * sample_label_mask
    #TODO check cudaerror: illegal memory access was encountered
    labels = labels * (1 - disable_mask) + disable_mask * ignore_label

    return labels

