import math

import numpy as np
import megengine as mge
import megengine.functional as F

from det_opr.utils import mask_to_inds

def roi_pool(rpn_fms, rois, stride, pool_shape, roi_type='roi_align', 
             labels=None, bbox_targets=None):
    assert len(stride) == len(rpn_fms)
    canonical_level = 4
    canonical_box_size = 224
    min_level = math.log2(stride[0])
    max_level = math.log2(stride[-1])

    num_fms = len(rpn_fms)
    box_sizes = F.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    level_assignments = F.floor(
	canonical_level + F.log(box_sizes / canonical_box_size) / np.log(2)
    )
    level_assignments = F.minimum(level_assignments, max_level)
    level_assignments = F.maximum(level_assignments, min_level)
    level_assignments = level_assignments - min_level
    available_masks = F.concat(
        [mge.ones(level_assignments.shapeof()[0]), mge.zeros(num_fms)], axis=0)
    level_assignments = F.concat([level_assignments, mge.tensor(np.arange(num_fms, dtype=np.int32))], axis=0)
    rois = F.concat([rois, mge.zeros((num_fms, rois.shapeof()[-1]))], axis=0)
    if labels is not None:
        labels = F.concat([labels, mge.ones((num_fms, labels.shapeof()[-1]))], axis=0)
        bbox_targets = F.concat([bbox_targets, mge.zeros((num_fms, bbox_targets.shapeof()[-1]))], axis=0)
    pool_list, inds_list = [], []
    for i in range(len(rpn_fms)):
        mask = level_assignments == i
        inds = mask_to_inds(mask)
        rois_fm = rois.ai[inds]
        if roi_type == 'roi_pool':
            pool_fm = F.roi_pooling(
                    rpn_fms[i], rois_fm, pool_shape, mode='max', scale=1.0/stride[i])
        elif roi_type == 'roi_align':
            pool_fm = F.roi_align(
                    rpn_fms[i], rois_fm, pool_shape, mode='average', 
                    spatial_scale=1.0/stride[i], sample_points=2, aligned=True)
        pool_list.append(pool_fm)
        inds_list.append(inds)

    fm_order = F.concat(inds_list, axis=0)
    pool_feature = F.concat(pool_list, axis=0)

    ordered_available_masks = available_masks.ai[fm_order]
    available_inds = mask_to_inds(ordered_available_masks)
    pool_feature = pool_feature.ai[available_inds]
    rois = rois.ai[fm_order, :].ai[available_inds, :]
    if labels is not None:
        labels = labels.ai[fm_order].ai[available_inds]
        bbox_targets = bbox_targets.ai[fm_order, :].ai[available_inds, :]
        return pool_feature, rois, F.zero_grad(labels), F.zero_grad(bbox_targets)
    else:
        return pool_feature, rois, None, None

