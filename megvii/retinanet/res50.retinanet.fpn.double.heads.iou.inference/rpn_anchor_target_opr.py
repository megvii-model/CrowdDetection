from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr
import torch
import numpy as np
import pdb

def _compute_center(boxes):

    ptrx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    ptry = 0.5 * (boxes[:, 1] + boxes[:, 3])
    centre = torch.stack([ptrx, ptry], dim=1)
    return centre

def _compute_pos_area(gtboxes, ratio = 0.3):

    H, W = gtboxes[:, 3] - gtboxes[:, 1], gtboxes[:, 2] - gtboxes[:, 0]
    centres = _compute_center(gtboxes)
    l = centres[:, 0] - ratio * W
    r = centres[:, 0] + ratio * W
    t = centres[:, 1] - ratio * H
    b = centres[:, 1] + ratio * H
    boundary = torch.stack([l, t, r, b], dim = 1)
    return boundary

def _anchor_double_target(gt_boxes, im_info, all_anchors):

    gt_boxes, im_info = gt_boxes.detach(), im_info.detach()
    all_anchors = all_anchors.detach()

    gt_boxes = gt_boxes[:im_info[5], :]
    dummy = -torch.ones([1, gt_boxes.shape[1]]).to(gt_boxes.device)
    gt_boxes = torch.cat([gt_boxes, dummy], dim=0)
    valid_mask = 1 - (gt_boxes[:, 4] < 0).float()

    anchor_centers = _compute_center(all_anchors)
    gtboxes_centers = _compute_center(gt_boxes)
    gtboxes_centers = gtboxes_centers * valid_mask.unsqueeze(1)

    N, K = all_anchors.shape[0], gt_boxes.shape[0]
    an_centers = anchor_centers.unsqueeze(1).repeat(1, K, 1)
    gt_centers = gtboxes_centers.unsqueeze(0).repeat(N, 1, 1)

    distance = torch.abs(an_centers - gt_centers)
    distance = torch.sqrt(torch.pow(distance, 2).sum(dim=2))
    
    start = 0
    end = 5
    overlaps = box_overlap_opr(all_anchors[:, :4], gt_boxes[:, :4])
    overlaps = overlaps * valid_mask.unsqueeze(0)
    default_num = 16

    ious_list = []

    for l in range(start, end):

        index = torch.nonzero(all_anchors[:, 4].eq(l), as_tuple=False)[:,0]
        level_dist = distance[index, :].permute(1, 0)
        ious = overlaps[index, :].permute(1, 0)
        sorted_index = torch.argsort(level_dist, 1, descending=False)
        n = min(sorted_index.shape[1], default_num)
        ious = torch.gather(ious, 1, sorted_index[:, :n]).permute(1, 0)

        ious_list.append(ious)

    ious = torch.cat(ious_list, dim=0)
    mean_var = ious.mean(0)
    std_var = ious.std(0)
    iou_thresh_per_gt = mean_var + std_var

    iou_thresh_per_gt = torch.clamp(iou_thresh_per_gt, 0.2)

    # limits the anchor centers in the gtboxes
    N, K = all_anchors.shape[0], gt_boxes.shape[0]
    anchor_points = an_centers
    pos_area = _compute_pos_area(gt_boxes, 0.3)
    pos_area = pos_area.unsqueeze(0).repeat(N, 1, 1)

    l = anchor_points[:, :, 0] - pos_area[:, :, 0]
    r = pos_area[:, :, 2] - anchor_points[:, :, 0]
    t = anchor_points[:, :, 1] - pos_area[:, :, 1]
    b = pos_area[:, :, 3] - anchor_points[:, :, 1]

    is_in_gt = torch.stack([l, r, t, b], dim=2)
    is_in_gt = is_in_gt.min(axis = 2)[0] > 0.1
    valid_mask = (overlaps >= iou_thresh_per_gt.unsqueeze(0)) * is_in_gt.float()
    ious = overlaps * valid_mask

    sorted_index = torch.argsort(ious, 1, descending=True)
    sorted_overlaps = torch.gather(ious, 1, sorted_index)
    max_overlaps = sorted_overlaps[:, :2].flatten()
    argmax_overlaps = sorted_index[:, :2].flatten()

    n, c = all_anchors.shape
    device = all_anchors.device
    labels = -torch.ones(2 * n).to(device)
    positive_mask = (max_overlaps >= 0.2).float().to(device)
    negative_mask = (max_overlaps < 0.2).float().to(device)
    labels = positive_mask + labels * (1 - positive_mask) * (1 - negative_mask)

    # bbox_targets = torch.zeros((n, 4)).to(device)
    bbox_targets = gt_boxes[argmax_overlaps, :4]
    all_anchors = all_anchors.unsqueeze(1).repeat(1, 2, 1).view(-1, c)
    bbox_targets = bbox_transform_opr(all_anchors[:, :4], bbox_targets)

    labels_cat = gt_boxes[argmax_overlaps, 4]
    labels_cat = labels_cat * (1 - labels.eq(0).float())
    labels_cat = labels_cat * (1 - labels.eq(-1).float()) - (labels.eq(-1).float())

    return labels, bbox_targets, labels_cat
    
def _anchor_target(gt_boxes, im_info, all_anchors):

    gt_boxes, im_info = gt_boxes.detach(), im_info.detach()
    all_anchors = all_anchors.detach()

    gt_boxes = gt_boxes[:im_info[5], :]
    valid_mask = 1 - (gt_boxes[:, 4] < 0).float()

    anchor_centers = _compute_center(all_anchors)
    gtboxes_centers = _compute_center(gt_boxes) * valid_mask.unsqueeze(1)

    N, K = all_anchors.shape[0], gt_boxes.shape[0]
    an_centers = anchor_centers.unsqueeze(1).repeat(1, K, 1)
    gt_centers = gtboxes_centers.unsqueeze(0).repeat(N, 1, 1)

    distance = torch.abs(an_centers - gt_centers)
    distance = torch.sqrt(torch.pow(distance, 2).sum(dim=2))
    
    start = 0
    end = 5
    overlaps = box_overlap_opr(all_anchors[:, :4], gt_boxes[:, :4])
    overlaps = overlaps * valid_mask.unsqueeze(0)
    default_num = 9

    ious_list = []
    for l in range(start, end):

        index = torch.nonzero(all_anchors[:,4].eq(l), as_tuple=False)[:, 0]
        level_dist = level_dist[index, :].permute(1, 0)
        ious = distance[index, :].permute(1, 0)
        sorted_index = torch.argsort(ious, 1, descending=False)
        n = min(default_num, sorted_index.shape[1])
        ious = torch.gather(ious, 1, sorted_index[:, :n]).permute(1, 0)
        ious_list.append(ious)

    ious = torch.cat(ious_list, dim=0)
    mean_var = ious.mean(0)
    std_var = ious.std(0)
    iou_thresh_per_gt = mean_var + std_var

    iou_thresh_per_gt = torch.clamp(iou_thresh_per_gt, 0.35)
    n = iou_thresh_per_gt.shape[0]

    # limits the anchor centers in the gtboxes
    N, K = all_anchors.shape[0], gt_boxes.shape[0]
    anchor_points = an_centers
    proxies = gt_boxes.unsqueeze(0).repeat(N, 1, 1)
    l = anchor_points[:, :, 0] - proxies[:, :, 0]
    r = proxies[:, :, 2] - anchor_points[:, :, 0]
    t = anchor_points[:, :, 1] - proxies[:, :, 1]
    b = proxies[:, :, 3] - anchor_points[:, :, 1]

    is_in_gt = torch.stack([l, r, t, b], dim=2)
    is_in_gt = is_in_gt.min(axis = 2) > 0.1
    valid_mask = (overlaps >= iou_thresh_per_gt.unsqueeze(0)) * is_in_gt
    ious = overlaps * valid_mask

    argmax_overlaps = torch.argmax(ious, dim=1)
    max_overlaps = torch.gather(ious, 1, argmax_overlaps.unsqueeze(1))

    n = all_anchors.shape[0]
    labels = -torch.ones(n)
    positive_mask = max_overlaps > 0
    negative_mask = max_overlaps < config.rpn_negative_overlap
    labels = positive_mask + labels * (1 - positive_mask) * (1 - negative_mask)

    bbox_targets = gt_boxes[argmax_overlaps, :4]
    bbox_targets = bbox_transform_opr(all_anchors[:, :4], bbox_targets)

    labels_cat = gt_boxes[argmax_overlaps, 4]
    labels_cat = labels_cat * (1 - labels.eq(0).float())
    labels_cat = labels_cat * (1 - labels.eq(-1).float()) - labels.eq(-1).float()

    return labels, bbox_targets, labels_cat
    
def rpn_anchor_target_opr(gt_boxes, im_info, anchors):

    rpn_label_list, rpn_target_boxes_list, iou_thresh_list = [], [], []
    for i in range(config.train_batch_per_gpu):

        rpn_labels, rpn_target_boxes, _ = _anchor_double_target(gt_boxes[i], im_info[i], anchors)
        rpn_labels = rpn_labels.reshape(-1, 2)
        c = rpn_target_boxes.shape[1]
        rpn_target_boxes = rpn_target_boxes.view(-1, 2, c)
        
        # mask the anchors overlapping with ignore regions
        ignore_label = mask_anchor_opr(gt_boxes[i], im_info[i], anchors, rpn_labels[:, 0])
        rpn_labels = rpn_labels - rpn_labels.eq(0).float() * (ignore_label < 0).unsqueeze(1).float()
        
        rpn_label_list.append(rpn_labels.unsqueeze(0))
        rpn_target_boxes_list.append(rpn_target_boxes.unsqueeze(0))

    rpn_labels = torch.cat(rpn_label_list, axis = 0)
    rpn_target_boxes = torch.cat(rpn_target_boxes_list, axis = 0)
    return rpn_labels, rpn_target_boxes

def mask_anchor_opr(gtboxes, im_info, anchors, labels):
    
    eps = 1e-6
    gtboxes = gtboxes[:im_info[5], :]
    ignore_mask = gtboxes[:, 4] < 0

    mask_flag = torch.zeros(labels.shape[0])
    N, K = anchors.shape[0], gtboxes.shape[0]
    p_pred = anchors.unsqueeze(1).repeat(1, K, 1)
    p_gt = gtboxes.unsqueeze(0).repeat(N, 1, 1)
    
    max_off = torch.cat([torch.max(p_pred[:,:, :2], p_gt[:,:,:2]),
                         torch.min(p_pred[:, :, 2:4], p_gt[:, :, 2:4])],
                         axis = 2)
   
    I = torch.clamp(max_off[:, :, 2] - max_off[:, :, 0] + 1, 0) * torch.clamp(
        max_off[:, :, 3] - max_off[:, :, 1] + 1, 0)
    A = (p_pred[:, :, 2] - p_pred[:, :, 0] + 1) * (p_pred[:, :, 3] - p_pred[:, :, 1] + 1)
    I = torch.clamp(I, 0)
    A = torch.clamp(A, 0)
    IoA = I / (A + eps)
    IoA = IoA * ignore_mask.unsqueeze(0)
    mask_flag = (IoA > 0.5).sum(axis=1) > 0

    labels = labels - labels.eq(0).float() * mask_flag
    return labels

def rpn_anchor_target_opr_impl(
        gt_boxes, im_info, anchors, clobber_positives = True, ignore_label=-1,
        background_label=0):
    

    gt_boxes, im_info = gt_boxes.detach(), im_info.detach()
    anchors = anchors.detach()

    # NOTE: For multi-gpu version, this function should be re-written
    a_shp0 = anchors.shape[0]
    valid_gt_boxes = gt_boxes[:im_info[5], :]
    valid_mask = (gt_boxes[:im_info[5], 4] > 0).float()
    overlaps = box_overlap_opr(anchors[:, :4], valid_gt_boxes[:, :4])
    overlaps = overlaps * valid_mask.unsqueeze(0)

    argmax_overlaps = torch.argmax(overlaps,dim=1)
    max_overlaps = torch.gather(overlaps, 1, argmax_overlaps.unsqueeze(1))
    gt_argmax_overlaps = torch.argmax(overlaps, dim=0)
    gt_argmax_overlaps = torch.gather(overlaps, 1, gt_argmax_overlaps.unsqueeze(0))

    cond_max_overlaps = overlaps.eq(gt_argmax_overlaps).float()
    cmo_shape1 = cond_max_overlaps.shape[1]

    gt_argmax_overlaps = torch.nonzero(cond_max_overlaps.flatten(), as_tuple=False)
    gt_argmax_overlaps = gt_argmax_overlaps // cmo_shape1

    labels = ignore_label * torch.ones(a_shp0)
    fg_mask = (max_overlaps >= config.rpn_positive_overlap).float()
    fg_mask[gt_argmax_overlaps] = 1
    index = torch.nonzero(fg_mask, as_tuple=False).view(-1).long()
    labels[index] = 1

    bbox_targets = bbox_transform_opr(anchors, valid_gt_boxes[index, :4])


    # fg_mask[gt_argmax_overlaps]

    # --- megbrain fashion code ---
    # argmax_overlaps = O.Argmax(overlaps, axis=1)
    # max_overlaps = O.IndexingOneHot(overlaps, 1, argmax_overlaps)
    # gt_argmax_overlaps = O.Argmax(overlaps, axis=0)
    # gt_max_overlaps = O.IndexingOneHot(overlaps, 0, gt_argmax_overlaps)

    # cond_max_overlaps = overlaps.eq(gt_max_overlaps.add_axis(0))
    # cmo_shape1 = cond_max_overlaps.shape[1]

    # gt_argmax_overlaps = \
    #     O.CondTake(cond_max_overlaps.flatten(), cond_max_overlaps.flatten(),
    #                'EQ',1).outputs[1]
    # # why should be divided by the cmo_shape1
    # gt_argmax_overlaps = gt_argmax_overlaps // cmo_shape1

    # labels = O.ones(a_shp0) * ignore_label
    # const_one = O.ConstProvider(1.0)
    # if not clobber_positives:
    #     labels = labels * (max_overlaps >= config.rpn_negative_overlap)

    # fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    # fg_mask = fg_mask.set_ai[gt_argmax_overlaps](
    #     const_one.broadcast(gt_argmax_overlaps.shape))

    # fg_mask_ind = O.CondTake(fg_mask, fg_mask, 'EQ', 1).outputs[1]
    # labels = labels.set_ai[fg_mask_ind](const_one.broadcast(fg_mask_ind.shape))

    # if clobber_positives:
    #     labels = labels * (max_overlaps >= config.rpn_negative_overlap)

    # Here, we compute the targets for each anchors
    # bbox_targets = bbox_transform_opr(
    #     anchors, valid_gt_boxes.ai[argmax_overlaps, :4])

    return labels, bbox_targets

