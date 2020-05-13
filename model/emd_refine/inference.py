import os
import math
import argparse

import cv2
import numpy as np
import megengine as mge
from megengine import jit

from config import config
import dataset
import network
import misc_utils

if_set_nms = True
top_k = 2

def inference(args):
    @jit.trace(symbolic=False)
    def val_func():
        pred_boxes = net(net.inputs)
        return pred_boxes
    # model path
    saveDir = config.model_dir
    evalDir = config.eval_dir
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir,
            'epoch_{}.pkl'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # load model
    net = network.Network()
    net.eval()
    check_point = mge.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    image, im_info = get_data(args.img_path)
    net.inputs["image"].set_value(image.astype(np.float32))
    net.inputs["im_info"].set_value(im_info)
    pred_boxes = val_func().numpy()
    num_tag = config.num_classes - 1
    target_shape = (pred_boxes.shape[0]//num_tag//top_k, top_k)
    pred_tags = (np.arange(num_tag) + 1).reshape(-1,1)
    pred_tags = np.tile(pred_tags, target_shape).reshape(-1,1)
    # nms
    if if_set_nms:
        from set_nms_utils import set_cpu_nms
        n = pred_boxes.shape[0] // top_k
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, -2] > 0.05
        pred_boxes = pred_boxes[keep]
        pred_tags = pred_tags[keep]
        keep = set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep][:, :-1]
        pred_tags = pred_tags[keep].flatten()
    else:
        from set_nms_utils import cpu_nms
        keep = pred_boxes[:, -1] > 0.05
        pred_boxes = pred_boxes[keep]
        pred_tags = pred_tags[keep]
        keep = cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
        pred_tags = pred_tags[keep].flatten()
    result_dict = dict(height=int(im_info[0, -2]), width=int(im_info[0, -1]),
        dtboxes=boxes_dump(pred_boxes, pred_tags))
    name = args.img_path.split('/')[-1].split('.')[-2]
    misc_utils.save_json_lines([result_dict], '{}.json'.format(name))

def get_data(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    transposed_img = np.ascontiguousarray(
        resized_img.transpose(2, 0, 1)[None, :, :, :],
        dtype=np.float32)
    im_info = np.array([height, width, scale, original_height, original_width],
        dtype=np.float32)[None, :]
    return transposed_img, im_info

def boxes_dump(boxes, pred_tags):
    result = []
    boxes = boxes.tolist()
    for idx in range(len(boxes)):
        box = boxes[idx]
        box_dict = {}
        box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        box_dict['tag'] = int(pred_tags[idx])
        box_dict['score'] = box[-1]
        result.append(box_dict)
    return result

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--img_path', '-i', default=None, type=str)
    args = parser.parse_args()
    inference(args)

if __name__ == '__main__':
    run_inference()
