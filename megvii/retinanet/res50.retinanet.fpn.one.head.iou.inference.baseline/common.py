#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import os.path as osp
import cv2, pickle
import json, random
from config import config
import time
from box import *
from matching import maxWeightMatching
from draw import *
from detbox import *
from infrastructure import *
from detToolkits.detools import *
from nms_wrapper import *
from tqdm import tqdm
from collections import Counter
from multiprocessing import Queue, Process
import pdb
def compute_mAP(dtpath):
    gtpath = config.eval_json
    dbName = 'human'
    db = EvalDB(dbName, '',gtpath, dtpath, None)
    DBs = [db]
    evaluator = Evaluator(DBs)
    mAP,_ = evaluator.eval_AP()
    mMR = evaluator.eval_MR()
    return mAP[0],mMR[0]
def recover_dtboxes(record):

    assert 'dtboxes' in record
    if len(record['dtboxes']) < 1:
        return np.empty([0, 5])
    dtboxes = np.vstack([np.hstack((rb['box'], rb['score'])) for rb in record['dtboxes']])
    dtboxes = recover_func(dtboxes)
    return dtboxes

def recover_rois(record):

    assert 'rois' in record
    if len(record['rois']) < 1:
        return np.empty([0, 5])
    rois = np.vstack([np.hstack((rb['box'], rb['score'])) for rb in record['rois']])
    rois = recover_func(rois)
    return rois

def recover_gtboxes(record):

    assert 'gtboxes' in record
    if len(record['gtboxes']) < 1:
        return np.empty([0, 4])
    gtboxes = np.vstack([rb['box'] for rb in record['gtboxes']])
    gtboxes = recover_func(gtboxes)
    return gtboxes

def compute_JI(dtboxes, gtboxes, bm_thr):

    assert dtboxes.shape[-1] > 3 and gtboxes.shape[-1] > 3
    eps = 1e-6
    matches = compute_JC(dtboxes, gtboxes, bm_thr)
    k = len(matches)
    m = gtboxes.shape[0]
    n = dtboxes.shape[0]
    ratio = k / (m + n -k + eps)
    recall = k / (m + eps)
    cover = k / (n + eps)
    noise = 1 - cover
    result_dict = dict(ratio = ratio, recall = recall, cover = cover,
        noise = noise, k = k, m = m, n = n)
    return result_dict
    
def get_ignores(indices, boxes, ignores, ioa_thr):

    indices = list(set(np.arange(boxes.shape[0])) - set(indices))
    rboxes = boxes[indices, :]
    ioas = compute_ioa_matrix(rboxes, ignores)
    ioas = np.max(ioas, axis = 1)
    rows = np.where(ioas > ioa_thr)[0]
    return rows.size

def worker(result_queue, records, gt, score_thr, bm_thr):

    total, eps = len(records), 1e-6
    for i in range(total):
        record = records[i]
        ID = record['ID']
        height, width = record['height'], record['width']

        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        GT = list(filter(lambda rb:rb['ID'] == ID, gt))
        if len(GT) < 1:
            result_queue.put_nowait(None)
            continue
        
        GT = GT[0]
        flags = np.array([is_ignore(rb) for rb in GT['gtboxes']])
        rows = np.where(~flags)[0]
        ignores = np.where(flags)[0]

        gtboxes = np.vstack([GT['gtboxes'][j]['fbox'] for j in rows])
        gtboxes = recover_func(gtboxes)
        gtboxes = clip_boundary(gtboxes, height, width)

        if ignores.size:
            ignores = np.vstack([GT['gtboxes'][j]['fbox'] for j in ignores])
            ignores = recover_func(ignores)
            ignores = clip_boundary(ignores, height, width)

        dtboxes = np.vstack([np.hstack([rb['box'], rb['score']]) for rb in record['dtboxes']])
        dtboxes = recover_func(dtboxes)
        dtboxes = clip_boundary(dtboxes, height, width)
        rows = np.where(dtboxes[:,-1]> score_thr)[0]
        dtboxes = dtboxes[rows,...]

        matches = compute_JC(dtboxes, gtboxes, bm_thr)
        dt_ign, gt_ign = 0, 0

        if ignores.size:
            indices = np.array([j for (j,_) in matches])
            dt_ign = get_ignores(indices, dtboxes, ignores, bm_thr)
            indices = np.array([j for (_,j) in matches])
            gt_ign = get_ignores(indices, gtboxes, ignores, bm_thr)

        k = len(matches)
        m = gtboxes.shape[0] - gt_ign
        n = dtboxes.shape[0] - dt_ign

        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover

        result_dict = dict(ID = ID, ratio = ratio, recall = recall , noise = noise ,
            cover = cover, k= k ,n = n, m = m)
        result_queue.put_nowait(result_dict)

def common_process(func, data, nr_procs, *args):

    total = len(data)
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    tqdm.monitor_interval = 0
    pbar = tqdm(total = total, leave = False, ascii = True)
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data[start:end]
        # func(result_queue, sample_data, *args)
        p = Process(target= func,args=(result_queue, sample_data, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return results

def is_ignore(record):

    flag = False
    if 'extra' in record:
        if 'ignore' in record['extra']:
            flag = True if record['extra']['ignore'] else False
    return flag

def strline(results):
    
    assert len(results)
    m = 4370
    mean_ratio = np.sum([rb['ratio'] for rb in results]) / m
    mean_cover = np.sum([rb['cover'] for rb in results]) / m
    mean_recall = np.sum([rb['recall'] for rb in results]) / m
    mean_noise = 1 - mean_cover
    valids = np.sum([rb['k'] for rb in results])
    total = np.sum([rb['n'] for rb in results])
    gtn = np.sum([rb['m'] for rb in results])

    line = 'mean_ratio:{:.4f}, mean_cover:{:.4f}, mean_recall:{:.4f}, mean_noise:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
        mean_ratio, mean_cover, mean_recall, mean_noise, valids, total, gtn)
    return line
def clip_boundary(dtboxes,height,width):

    assert dtboxes.shape[-1]>=4
    dtboxes[:,0] = np.minimum(np.maximum(dtboxes[:,0],0), width - 1)
    dtboxes[:,1] = np.minimum(np.maximum(dtboxes[:,1],0), height - 1)
    dtboxes[:,2] = np.maximum(np.minimum(dtboxes[:,2],width), 0)
    dtboxes[:,3] = np.maximum(np.minimum(dtboxes[:,3],height), 0)
    return dtboxes

def ensure_dir(dirpath):

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
def box_dump(pboxes):

    pboxes = pboxes[:, :5] if pboxes.shape[-1] > 5 else pboxes
    total = pboxes.shape[0]
    if pboxes.shape[-1]==4:
        pboxes = np.hstack([pboxes,np.ones([total,1],dtype=np.float64)])
        
    boxes = []
    for i in range(total):
        db = np.float64(pboxes[i,:])
        dbox = DetBox(db[0],db[1],db[2]-db[0],db[3]-db[1],tag = 1,score = db[4])
        boxes.append(dbox.dumpOdf())
    return boxes

def dump_detection(dtboxes):

    assert dtboxes.shape[1] > 4
    m = dtboxes.shape[0]
    results = []
    for i in range(m):
        db = np.float64(dtboxes[i, ...])
        dbox = {}
        dbox['box'] = [db[0], db[1], db[2] - db[0], db[3] - db[1]]
        dbox['score'] = db[4]
        dbox['tag'] = 1
        dbox['gene'] = db[5]
        results.append(dbox)

    return results

def save_results(content,fpath):

    with open(fpath,'w') as fid:
        for db in content:
            line = json.dumps(db)+'\n'
            fid.write(line)

def load_func(fpath):

    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def recover_func(bboxes):

    assert bboxes.shape[1]>=4
    bboxes[:, 2:4] += bboxes[:,:2]
    return bboxes
