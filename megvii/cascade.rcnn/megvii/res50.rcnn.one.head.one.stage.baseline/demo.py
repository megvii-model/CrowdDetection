from config import config
import os.path as osp
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

def computeIoUs(fpath):
    
    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    mAP, mMR = compute_mAP(fpath)

    fid = open('results.md', 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path='results.md')

def eval_all():
    for epoch in range(25, 50):
        fpath = osp.join(config.eval_dir, 'epoch-{}.human'.format(epoch))
        if not os.path.exists(fpath):
            continue
        computeIoUs(fpath)

if __name__ == '__main__':
    eval_all()
