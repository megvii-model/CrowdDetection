import os
import math
import argparse
from multiprocessing import Process, Queue

from tqdm import tqdm
import numpy as np
import megengine as mge
from megengine import jit

from config import config
import network
import dataset
import misc_utils

def eval_all(args):
    # model_path
    saveDir = config.model_dir
    evalDir = config.eval_dir
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 
            'epoch_{}.pkl'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # load data
    records = misc_utils.load_json_lines(config.eval_source)
    # multiprocessing
    num_records = len(records)
    num_devs = args.devices
    num_image = math.ceil(num_records / num_devs)
    result_queue = Queue(1000)
    procs = []
    all_results = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, num_records)
        split_records = records[start:end]
        proc = Process(target=inference, args=(
                model_file, i, split_records, result_queue))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total=num_records, ncols=50)
    for i in range(num_records):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)
    for p in procs:
        p.join()
    fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    misc_utils.save_json_lines(all_results, fpath)

def inference(model_file, device, records, result_queue):
    @jit.trace(symbolic=False)
    def val_func():
        pred_boxes = net(net.inputs)
        return pred_boxes
    net = network.Network()
    net.eval()
    check_point = mge.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    for record in records:
        np.set_printoptions(precision=2, suppress=True)
        net.eval()
        image, gt_boxes, im_info, ID = get_data(record, device)
        net.inputs["image"].set_value(image.astype(np.float32))
        net.inputs["im_info"].set_value(im_info)
        pred_boxes = val_func().numpy()
        num_tag = config.num_classes - 1
        target_shape = (pred_boxes.shape[0]//num_tag, 1)
        pred_tags = (np.arange(num_tag) + 1).reshape(-1,1)
        pred_tags = np.tile(pred_tags, target_shape).reshape(-1,1)
        # nms
        from set_nms_utils import cpu_nms
        keep = pred_boxes[:, -1] > 0.05
        pred_boxes = pred_boxes[keep]
        pred_tags = pred_tags[keep]
        keep = cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
        pred_tags = pred_tags[keep].flatten()
        result_dict = dict(ID=ID, height=int(im_info[0, -2]), width=int(im_info[0, -1]),
                dtboxes=boxes_dump(pred_boxes, pred_tags, False),
                gtboxes=boxes_dump(gt_boxes, None, True))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes, pred_tags, is_gt):
    result = []
    boxes = boxes.tolist()
    for idx in range(len(boxes)):
        box = boxes[idx]
        if is_gt:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
        else:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = int(pred_tags[idx])
            box_dict['score'] = box[-1]
        result.append(box_dict)
    return result

def get_data(record, device):
    data = dataset.val_dataset(record)
    image, gt_boxes, ID = \
                data['data'], data['boxes'], data['ID']
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
    return transposed_img, gt_boxes, im_info, ID

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--devices', '-d', default=1, type=int)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_test()

