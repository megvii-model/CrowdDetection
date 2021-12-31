import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
from config import *
from network import Network
from CrowdHuman import CrowdHuman
from utils import nms_utils
from nms_wrapper import nms
from utils.misc_utils import *
from utils.nms_utils import emd_cpu_nms
import multiprocessing as mp
import pdb
def eval_all(args, config, network, model_file, devices):
    
    crowdhuman = CrowdHuman(config, if_train=False)
    num_devs = len(devices)

    len_dataset = len(crowdhuman)
    num_image = math.ceil(len_dataset / num_devs)
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

@torch.no_grad()
def inference(config, network, model_file, device, dataset, start, end, result_queue):
    
    # init model
    net = network()
    check_point = torch.load(model_file,map_location='cpu')
    net.load_state_dict(check_point['state_dict'])
    net.cuda(device)
    net.eval()
    # init data
    dataset = dataset.records[start:end]
    crowdhuman = CrowdHuman(config, False, split_data = dataset)
    data_iter = torch.utils.data.DataLoader(dataset=crowdhuman, shuffle=False, num_workers=2)
    # inference
    for i, t in enumerate(data_iter):
        # pdb.set_trace()
        image, gt_boxes, im_info, ID = t
        pred_boxes = net(image.cuda(device), im_info.cuda(device))
        pred_boxes = pred_boxes.data.cpu().numpy()
        assert pred_boxes.shape[0] < 2
        pred_boxes = pred_boxes[0]
        gt_boxes = gt_boxes[0].data.cpu().numpy()
        scale = im_info[0, 2]

        
        if config.test_nms_method == 'set_nms':

            pred_boxes[:,:4] /= scale
            n = pred_boxes.shape[0] // 2
            ind = np.tile(np.arange(n).reshape(-1, 1), (1, 2)).reshape(-1, 1)

            scores = pred_boxes[:,4:6].prod(axis=1).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes[:,:4],scores, ind))
            flag = pred_boxes[:, 4] >= config.pred_cls_threshold
            pred_boxes = pred_boxes[flag]
            # keep = nms(np.float32(pred_boxes), 0.5)
            keep = emd_cpu_nms(pred_boxes, 0.5, 1)
            n = len(keep)
            tag = np.ones([n, 1])
            pred_boxes = np.hstack([pred_boxes[keep], tag])

        elif config.test_nms_method == 'normal_nms':

            assert pred_boxes.ndim == 2
            pred_boxes[:,:4] /= scale
            scores = pred_boxes[:, 4:6].prod(axis=1).reshape(-1, 1)
            pred_boxes = np.hstack([pred_boxes[:,:4], scores])
            flag = pred_boxes[:, 4] >= config.pred_cls_threshold
            cls_boxes = pred_boxes[flag]
            keep = nms(np.float32(cls_boxes), config.test_nms)
            pred_boxes = cls_boxes[keep]

        elif config.test_nms_method == 'none':

            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        
        else:
            raise ValueError('Unknown NMS method.')
     
        pred_boxes = xyxy2xywh(pred_boxes)
        gt_boxes = xyxy2xywh(gt_boxes)
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes):

    n, c = boxes.shape
    assert c > 3
    boxes = boxes.astype(np.float64)
    results = []
    for i in range(n):

        box = boxes[i]
        if c > 5:
            box, score, tag = box[:4], box[4], box[5]
        else:
            box, tag, score = box[:4], box[4], 1
        box_dict = {'box': [v for v in box], 'tag': tag, 'score': score}
        results.append(box_dict)
    return results

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
    mp.set_start_method('spawn')
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

