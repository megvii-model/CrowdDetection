from common import *
def computeJaccard(fpath, save_path ='results.md'):

    assert os.path.exists(fpath)
    records = load_func(fpath)

    GT = load_func(config.eval_json)
    fid = open(save_path, 'a')
    for i in range(10):
        score_thr = 1e-1 * i
        results = common_process(worker, records, 20, GT, score_thr, 0.5)
        line = strline(results)
        line = 'score_thr:{:.3f}, '.format(score_thr) + line
        print(line)
        fid.write(line + '\n')
        fid.flush()
    fid.close()

def filter_boxes(result_queue, records, score_thr):

    assert score_thr > 0
    for i, record in enumerate(records):
        
        boxes = list(filter(lambda rb: rb['score']>= score_thr, record['dtboxes']))
        if len(boxes) < 1:
            result_queue.put_nowait(None)
            continue
        record['dtboxes'] = boxes
        result_queue.put_nowait(record)

def compute_iou_worker(result_queue, records, score_thr):

    for record in records:
        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        gtboxes = recover_gtboxes(record)
        dtboxes = recover_dtboxes(record)
        result = compute_JI(dtboxes, gtboxes, 0.5)
        result['dtboxes'] = box_dump(dtboxes)
        result['ID'] = record['ID']
        result['height'] = record['height']
        result['width'] = record['width']
        result['gtboxes'] = record['gtboxes']

        result_queue.put_nowait(result)

def computeIoUs(fpath, score_thr = 0.05):
    
    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    records = load_func(fpath)
    results = common_process(filter_boxes, records, 16, score_thr)
    
    fpath = 'msra.human'
    save_results(results, fpath)
    mAP, mMR = compute_mAP(fpath)

    fid = open('record.txt', 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path='record.txt')

def test_unit():

    fpath = osp.join(config.eval_dir, 'epoch-30.human')
    records = load_func(fpath)
    save_path = 'nms.md'
    
    for i in range(1, 10):
        score_thr = 0.1 * i
        results = common_process(filter_boxes, records, 16, score_thr)
        fpath = 'mountain.human'
        save_results(results, fpath)
        mAP, mMR = compute_mAP(fpath)
        line = 'score_thr:{:.1f}, mAP:{:.4f}, mMR:{:.4f}'.format(score_thr, mAP, mMR)
        print(line)
        fid = open(save_path, 'a')
        fid.write(line + '\n')
        fid.close()
        computeJaccard(fpath, save_path)

def eval_all():
<<<<<<< HEAD
    for epoch in range(25, 60):
=======

    for epoch in range(10, 51):
>>>>>>> 126a572b4b37b562f2a6f27ee126688c37d13cf9
        fpath = osp.join(config.eval_dir, 'epoch-{}.human'.format(epoch))
        if not os.path.exists(fpath):
            continue
        computeIoUs(fpath)

def _distinguish_occ(gtboxes):

    overlaps = compute_iou_matrix(gtboxes, gtboxes)
    ious = np.triu(overlaps, 1).max(axis=1)
    flag = ious >= 0.5
    if flag.sum() < 1:
        gtboxes = np.hstack((gtboxes, np.zeros((gtboxes.shape[0], 1))))
        return gtboxes
    occs = gtboxes[flag]
    candidates = gtboxes[~flag]
    overlaps = compute_iou_matrix(candidates, occs)
    
    ious = overlaps.max(axis=1)
    flag = ious >= 0.5
    occs = np.vstack((occs, candidates[flag]))
    sparse = candidates[~flag]
    occs = np.hstack((occs, np.ones([occs.shape[0], 1])))
    sparse = np.hstack((sparse, np.zeros([sparse.shape[0], 1])))
    gtboxes = np.vstack((occs, sparse))
    return gtboxes

def detail_analysis():

    fpath = config.eval_source
    records = load_func(fpath)
    total = len(records)

    fpath = osp.join(config.eavl_dir, 'epoch-33.human')
    C = load_func(fpath)
    score_thr = 0.8
    results = []

    tqdm.monitor_interval = 0
    pbar = tqdm(total = total, leave = False, ascii = True)
    for record in records:

        name = record['ID']

        x = list(filter(lambda rb:rb['ID'] == name, C))
        if len(x) < 1:
            pbar.update(1)
            continue
        dtboxes = recover_dtboxes(x[0])        
        rows = np.where(dtboxes[:, 4] >= score_thr)[0]
        if rows.size < 1:
            pbar.update(1)
            continue
        dtboxes = dtboxes[rows, ...]
        height, width = record['height'], record['width']

        dtboxes = clip_boundary(dtboxes, height, width)
        gtboxes = recover_gtboxes(record)

        flags = np.array([is_ignore(rb) for rb in record['gtboxes']])
        rows = np.where(flags == 0)[0]
        if rows.size < 1:
            pbar.update(1)
            continue

        ignores = np.empty([0, gtboxes.shape[1]])  
        if flags.sum():
            ignores = gtboxes[flags > 0, ...]

        gtboxes = clip_boundary(gtboxes[rows, ...], height, width)
        gtboxes = _distinguish_occ(gtboxes)

        matches = compute_JC(dtboxes, gtboxes, 0.5)
        if len(matches) < 1:
            pbar.update(1)
            continue

        dt_ign, gt_ign = 0, 0
        if ignores.shape[0]:
            indices = np.array([j for (j,_) in matches])
            dt_ign = get_ignores(indices, dtboxes, ignores, 0.5)
            indices = np.array([j for (_,j) in matches])
            gt_ign = get_ignores(indices, gtboxes, ignores, 0.5)

        index = np.array([i for (_, i) in matches])
        flags = gtboxes[index, 4]
        rows = np.where(flags == 1)[0]
        cols = np.where(flags == 0)[0]
    
        m, n = dtboxes.shape[0] - dt_ign, gtboxes.shape[0] - gt_ign
        k = len(matches)
        ratio = k/(m + n - k + 1e-6)

        result = {}
        result['ID'] = name
        result['crowd'] = rows.size
        result['sparse'] = cols.size
        result['ratio'] = ratio
        results.append(result)
        pbar.update(1)

    crowd = np.sum([rb['sparse'] for rb in results])
    sparse = np.sum([rb['crowd'] for rb in results])
    ratio = np.sum([rb['ratio'] for rb in results]) / 4370
    line = 'sparse:{}, crowd:{}, total:{}, ratio:{:.4f}'.format(sparse, crowd, sparse + crowd, ratio)
    print(line)

if __name__ == '__main__':

    eval_all()
