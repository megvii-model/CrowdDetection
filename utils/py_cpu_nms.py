# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    eps = 1e-8
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
def _test():
    box1 = np.array([33,45,145,230,0.7])[None,:]
    box2 = np.array([44,54,123,348,0.8])[None,:]
    box3 = np.array([88,12,340,342,0.65])[None,:]
    boxes = np.concatenate([box1,box2,box3],axis = 0)
    nms_thresh = 0.5
    keep = py_cpu_nms(boxes,nms_thresh)
    alive_boxes = boxes[keep]
if __name__=='__main__':
    _test()