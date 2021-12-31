#!/usr/bin/python2
import argparse
from detools import *
import pdb
def nmsRule(a, b, nmsThres=0.65):
    return a.iomin(b) > nmsThres

def filterRule(a, scoreThres=0.0):
    return a.score > scoreThres

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', dest='dt', nargs='+', required=True, \
                        help='the fpaths to *.oddet (detection results)')
    parser.add_argument('--showThr', dest='showThr', nargs='+', default=[None], type=float, \
                        help='the thresholds for drawing detBoxes')
    parser.add_argument('--filterThr', dest='filterThr', nargs='+', default=[None], type=float, \
                        help='the thresholds for filtering detBoxes')
    parser.add_argument('--gt', dest='gt', required=True, \
                        help='the fpath to *.odgt (groundtruth)')
    parser.add_argument('--show', dest='show', default='all', \
                        help='image show options [all, fp, miss] or watchlist path')
    parser.add_argument('--axis', dest='axis', choices=[0,1], default=1, type=int, \
                        help='concate axis for multiple dts')
    parser.add_argument('--maxsize', dest='maxsize', default=640, type=int, \
                        help='maxsize for the show images')
    parser.add_argument('--which', dest='which', default=0, type=int, \
                        help='show detBoxes in which dt ranking')
    parser.add_argument('--nms', dest='nms', action='store_true', default=False, \
                        help='do nms or not')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, \
                        help='do shuffle or not for show results')
    args = parser.parse_args()
    gtpath, dtpaths = args.gt, args.dt
    doNMS, filterThres, showThres = args.nms, args.filterThr, args.showThr
    which, showopt, doShuffle = args.which, args.show, args.shuffle
    axis, maxsize = args.axis, args.maxsize

    if len(showThres) == 1:
        showThres = [showThres[0]] * len(dtpaths)
    else:
        assert len(showThres) == len(dtpaths), 'Num of showThrs must be equal to the num of dts.'

    if len(filterThres) == 1:
        filterThres = [filterThres[0]] * len(dtpaths)
    else:
        assert len(filterThres) == len(dtpaths), 'Num of filterThrs must be equal to the num of dts.'

    DBs = list()
    for (dtpath, showThr, filterThr) in zip(dtpaths, showThres, filterThres):
        dbName = dtpath.split('/')[-1]
        print('Loading {}...'.format(dbName))
        db = EvalDB(dbName, gtpath, dtpath, showThr)
        if filterThr is not None:
            print('Doing filtering for {}...'.format(dbName))
            db.filterBoxes('dt', filterRule, filterThr)
        if doNMS:
            print('Doing nms for {}...'.format(dbName))
            db.doNMS(nmsRule)
        DBs.append(db)
    
    print('Evaluating...')
    evaluator = Evaluator(DBs)
    evaluator.eval_MR()
    evaluator.eval_AP()
    showlist = evaluator.getShowlist(DBs[which].dbName, option=showopt)

    # displayer = Displayer(DBs)
    # displayer.show(showlist, doShuffle, axis, maxsize)

if __name__ == '__main__':
    main()
