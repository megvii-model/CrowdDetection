import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
import os, sys
import os.path as osp
from config import config
from network import Network
from data.CrowdHuman import CrowdHuman
from torch.utils.data import DataLoader
from utils.misc_utils import ensure_dir, device_parser, save_json_lines
from utils.nms_utils import set_cpu_nms as emd_cpu_nms
from nms_wrapper import nms
import torch.multiprocessing as mp
import pdb
def eval_all(args, config, network, model_file, devices):
    
    crowdhuman = CrowdHuman(config, if_train=False)
    num_devs = len(devices)
    len_dataset = len(crowdhuman)
    num_image = math.ceil(len_dataset / num_devs)
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue(5000)
    procs = []
    all_results = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, len_dataset)
        # inference(config, network, model_file, devices[i], crowdhuman, start, end, result_queue)
        proc = mp.Process(target=inference, args=(
                config, network, model_file, devices[i], crowdhuman, start, end, result_queue))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total = len_dataset, leave = False, ascii = True)
    for i in range(len_dataset):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)
    pbar.close()
    for p in procs:
        p.join()

    return all_results


def xyxy2xywh(boxes):

    assert boxes.shape[1] > 3
    boxes[:, 2:4] -= boxes[:, :2]
    return boxes

def inference(config, network, model_file, device, dataset, start, end, result_queue):
    
    torch.set_default_tensor_type('torch.FloatTensor')
    # init model
    net = network()

    # net = net.eval()
    check_point = torch.load(model_file, map_location='cpu')
    net.load_state_dict(check_point['state_dict'])
    net.cuda(device)
    net = net.eval()
    # init data
    # dataset.records = dataset.records[start:end]
    # splitted_data = dataset
    dataset = dataset.records[start:end]
    crowdhuman = CrowdHuman(config, if_train=False, splitted_data=dataset)
    data_iter = DataLoader(crowdhuman, shuffle = False,
            batch_size=1,
            num_workers=4,)
    # data_iter = .DataLoader(dataset=dataset, shuffle=False)
    # inference
    for i, t in enumerate(data_iter):

        image, gt_boxes, im_info, ID = t
        pred_boxes = net(image.cuda(device), im_info.cuda(device))
        del image, t
        pred_boxes = pred_boxes[:, 1]
        scale = im_info[0, 2]
        
        if config.test_nms_method == 'set_nms':
            assert pred_boxes.shape[1] > 4, "Not EMD Network! Using normal_nms instead."
            top_k = 2
            n, _ = divmod(pred_boxes.shape[0], top_k)
            pred_boxes[:, :4] /= scale
            idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
            tag = np.ones([pred_boxes.shape[0], 1])
            pred_boxes = np.hstack((pred_boxes[:, :5], idents, tag))
            flag = pred_boxes[:, 4] >= config.pred_cls_threshold
            pred_boxes = pred_boxes[flag]
            keep = emd_cpu_nms(pred_boxes, 0.5, 1.)
            n = len(keep)
            pred_boxes = pred_boxes[keep]
            tag = np.ones([n, 1])
            pred_boxes = np.hstack([pred_boxes[:, :5], tag])

        elif config.test_nms_method == 'normal_nms':

            pred_boxes[:, :4] /= scale
            n = pred_boxes.shape[0]
            tag = np.ones([n, 1])
            pred_boxes = np.hstack([pred_boxes, tag])
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep].astype(np.float32)
            keep = nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]
            tag = np.ones([len(keep), 1])
            pred_boxes = np.hstack([pred_boxes[:, :5], tag])
        elif config.test_nms_method == 'none':
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        else:
            raise NotImplementedError('Unknown NMS method.')

        pred_boxes = xyxy2xywh(pred_boxes)
        gt_boxes = xyxy2xywh(gt_boxes[0].numpy())
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes):
    if boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]
    elif boxes.shape[-1] == 6:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]

    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

def run_test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--start_epoch', '-s', default=25, type=int)
    parser.add_argument('--end_epoch','-e', default=35, type=int)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()
    
    saveDir = config.eval_dir
    ensure_dir(saveDir)
    devices = device_parser(args.devices)

    start_epoch, end_epoch = args.start_epoch, args.end_epoch
    for epoch in range(start_epoch, end_epoch):
        model_file = osp.join(config.model_dir, 'model-{}.pth'.format(epoch))
        if not osp.exists(model_file):
            continue

        results = eval_all(args, config, Network, model_file, devices)

        fpath = osp.join(saveDir, 'epoch-{}.human'.format(epoch))
        save_json_lines(results, fpath)

if __name__ == '__main__':
    run_test()

