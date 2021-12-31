from .database import *
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
class DisplayInfo(object):
    """
    :class: bbox and image information for display
    :ivar dtbox: the current detection result
    :ivar tag: the tag of current detection result, e.g., "Fp", "Miss"
    :ivar imgID: the current imageID
    :ivar pos: the rank position of current detection result 
    :ivar n: the total number of all detection results
    :type dtbox: DetBox
    :type tag: str
    :type imgID: str
    :type pos: int
    :type n: int
    """
    def __init__(self, imgID, tag=None, dtbox=None, pos=None, n=None):
        self.imgID = imgID
        self.dtbox = dtbox
        self.tag = tag
        self.pos = pos
        self.n = n

    def sprint(self):
        """
        :meth: convert into string
        """
        tagStr = "None" if self.tag is None else self.tag
        dtboxStr = "[None]" if self.dtbox is None else \
                    "[{}, {}, {}, {}, {}]".format(int(self.dtbox.x), int(self.dtbox.y), \
                                                  int(self.dtbox.w), int(self.dtbox.h), \
                                                  self.dtbox.score)

        if self.pos is not None and self.n is not None:
            rankStr = "{}/{}".format(self.pos, self.n)
        elif self.pos is not None:
            rankStr = "{}/None".format(self.pos)
        else:
            rankStr = "None/None"
        return tagStr + ", " + rankStr + ", " + dtboxStr + ", " + self.imgID 


class Evaluator(object):
    """
    :class: evaluator for multiple algorithms
    :ivar DBs: the detection results as well as groundtruth (stored in EvalDB) of all algorithms, using EvalDB.dbName as the key
    :ivar splitDBs: the splitted databases for self.DBs, using EvalDB.dbName as the key
    :ivar n: the number of EvalDBs
    :ivar colors: the colors for drawing curves
    :ivar results: the evaluation results of all EvalDBs, using EvalDB.dbName as the key
    :type DBs: dict
    :type splitDBs: dict
    :type n: int
    :type colors: list
    :type results: dict
    """
    def __init__(self, databases):
        self.DBs = dict()

        if type(databases) == EvalDB:
            self.DBs[databases.dbName] = databases
        else:
            for db in databases:
                self.DBs[db.dbName] = db

        self.n = len(self.DBs)
        self.colors = [(max(0.3,78*(i+1)%255/255.0), max(0.3,121*(i+1)%255/255.0),\
                        max(0.3,42*(i+1)%255/255.0)) for i in range(self.n)]
        self.results = None
        self.splitDBs = None

    def _splitClass(self):
        if self.splitDBs is None:
            self.splitDBs = dict()
            for dbName in self.DBs:
                self.splitDBs[dbName] = self.DBs[dbName].splitClass()

    def eval_MR(self, iouThres=0.5, matching=None, ref="CALTECH_-2", splitClass=False):
        """
        :meth: match the detection results with the groundtruth by iou measure for all algorithms, and draw fppi-miss and recall-precision curves 
        :param iouThres: iou threshold, default 0.5
        :param matching: match strategy, default None (Caltech matching strategy)
        :param ref: anchor points for calculating log-average miss rate, default "CALTECH_-2"
        :param splitClass: split database by class and evaluate each independently, default False
        :type iouThres: float
        :type matching: str - None/"VOC"
        :type ref: str - "CALTECH_-2"/"CALTECH_-4"
        :type splitClass: bool
        """
        self.results = dict()
        
        mMRv = []
        if splitClass is False: # evaluate over all
            for dbName in self.DBs:
                self.DBs[dbName].compare(iouThres, matching)
                self.results[dbName] = self.DBs[dbName].eval_MR(ref)

                mMRv.append(self.results[dbName][0])

            # plot curves
            # plt.figure("fppi-missrate result") 
            # plotlines = list()
            # legends = list()
            # for i, dbName in enumerate(self.results):
            #     fppi, missrate = self.results[dbName][1]
            #     plotline, = plt.semilogx(fppi, missrate, color=self.colors[i])
            #     plotlines.append(plotline)
            #     legends.append("{} {:.3f}".format(dbName, self.results[dbName][0]))
            # plt.legend(plotlines, legends)
            # plt.xlabel("false positive per image")
            # plt.ylabel("miss rate")
            # plt.show()
        else: # evaluate over each class
            self._splitClass()
            for dbName in self.splitDBs:
                mMR = 0.0
                MRs = dict()
                nCls = 0
                for cls in self.splitDBs[dbName]:
                    if cls <= 0:
                        continue
                    tmpDB = self.splitDBs[dbName][cls]
                    tag = tmpDB.mapClass2Tag(cls)
                    tmpDB.compare(iouThres, matching)
                    MR, curve = tmpDB.eval_MR(ref)
                    mMR += MR
                    MRs[tag] = MR
                    nCls += 1
                self.results[dbName] = [mMR/nCls, MRs]
                mMRv.append(self.results[dbName][0])
                # print(dbName, "mMR = {}".format(self.results[dbName][0]))
                # print(self.results[dbName][1])
        return mMRv

    def eval_AP_detail(self, iouThres=0.5, matching=None, metric=None, splitClass=False, csvname=None):
        """
        :meth: match the detection results with the groundtruth by iou measure for all algorithms, and draw fppi-miss and recall-precision curves 
        :param iouThres: iou threshold, default 0.5
        :param matching: match strategy, default None (Caltech matching strategy)
        :param metric: metric for calculating AP, default None (general AP calculation)
        :param splitClass: split database by class and evaluate each independently, default False
        :type iouThres: float
        :type matching: str - None/"VOC"
        :type metric: str - None/"VOC"
        :type splitClass: bool
        """
        
        self.results = dict()
        if splitClass is False: # evaluate over all
            for dbName in self.DBs:
                self.DBs[dbName].compare(iouThres, matching)
                self.results[dbName] = self.DBs[dbName].eval_AP_detail(metric)
                print(dbName, "mAP = {}".format(self.results[dbName][0]))

                curve = self.results[dbName][1]
                total_dets = len(self.results[dbName][1][0])
                total_gts = self.DBs[dbName].getGtNum() - self.DBs[dbName].getIgnNum() 
                print("FP: {}/{}, recall: {}/{}:{}, images: {}\n".format((1-curve[1][-1])*total_dets, total_dets, \
                                                           curve[0][-1]*total_gts, total_gts, curve[0][-1], len(self.DBs[dbName].images)))
                
            def dump_curve_csv(curve, filename=None):
                if filename == None:
                    fres = open( 'res.csv', 'w')
                else:
                    fres = open(filename, 'w')
                fres.write('threshold, fp, recall, fppi, recallrate, precision\n')
                for r, p, thr, fpn, recalln, fppi in zip(*curve):
                    fres.write( '{},{},{},{},{},{}\n'.format(thr, fpn, recalln, fppi, r, p))
                fres.close()
        return all_mAP


    def eval_AP(self, iouThres=0.5, matching=None, metric=None, splitClass=False, noshow=True):
        """
        :meth: match the detection results with the groundtruth by iou measure for all algorithms, and draw fppi-miss and recall-precision curves 
        :param iouThres: iou threshold, default 0.5
        :param matching: match strategy, default None (Caltech matching strategy)
        :param metric: metric for calculating AP, default None (general AP calculation)
        :param splitClass: split database by class and evaluate each independently, default False
        :type iouThres: float
        :type matching: str - None/"VOC"
        :type metric: str - None/"VOC"
        :type splitClass: bool
        """
        self.results = dict()
        all_mAP = []
        if splitClass is False: # evaluate over all
            for dbName in self.DBs:

                self.DBs[dbName].compare(iouThres, matching)
                self.results[dbName] = self.DBs[dbName].eval_AP_detail(metric)

                all_mAP.append(self.results[dbName][0])
                curve = self.results[dbName][1]
                total_dets = len(self.results[dbName][1][0])
                total_gts = self.DBs[dbName].getGtNum() - self.DBs[dbName].getIgnNum() 


            # if not noshow:
            #     # plot curves
            #     plt.figure("recall-precision result") 
            #     plotlines = list()
            #     legends = list()
            #     for i, dbName in enumerate(self.results):
            #         #recall, precision = self.results[dbName][1]
            #         recall = self.results[dbName][1][0]
            #         precision = self.results[dbName][1][1]
            #         plotline, = plt.plot(recall, precision, color=self.colors[i])
            #         plotlines.append(plotline)
            #         legends.append("{} {:.3f}".format(dbName, self.results[dbName][0]))
            #     plt.legend(plotlines, legends, loc="lower left")
            #     plt.xlabel("recall")
            #     plt.ylabel("precision")
            #     #plt.show()
            # else:
            return all_mAP,(int((1-curve[1][-1])*total_dets), total_dets, \
                          curve[0][-1]*total_gts, total_gts, curve[0][-1], len(self.DBs[dbName].images))
 
        else: # evaluate over each class
            self._splitClass()
            for dbName in self.splitDBs:
                mAP = 0.0
                APs = dict()
                nCls = 0
                for cls in self.splitDBs[dbName]:
                    if cls <= 0:
                        continue
                    tmpDB = self.splitDBs[dbName][cls]
                    tag = tmpDB.mapClass2Tag(cls)
                    tmpDB.compare(iouThres, matching)
                    AP, curve = tmpDB.eval_AP(metric)
                    mAP += AP
                    APs[tag] = AP
                    nCls += 1
                self.results[dbName] = [mAP/nCls, APs]
                print(dbName, "mAP = {}".format(self.results[dbName][0]))
                print(self.results[dbName][1])
    def eval_AP_condition(self, iouThres=0.5,score_thresh=0.1, matching=None, 
        metric=None, splitClass=False, noshow=True):
        """
        :meth: match the detection results with the groundtruth by iou measure for all algorithms, and draw fppi-miss and recall-precision curves 
        :param iouThres: iou threshold, default 0.5
        :param matching: match strategy, default None (Caltech matching strategy)
        :param metric: metric for calculating AP, default None (general AP calculation)
        :param splitClass: split database by class and evaluate each independently, default False
        :type iouThres: float
        :type matching: str - None/"VOC"
        :type metric: str - None/"VOC"
        :type splitClass: bool
        """
        self.results = dict()
        all_mAP = []
        if splitClass is False: # evaluate over all
            for dbName in self.DBs:
                self.DBs[dbName].compare(iouThres, matching)
                self.DBs[dbName].filter(score_thresh)
                self.results[dbName] = self.DBs[dbName].eval_AP_detail(metric)
                
                all_mAP.append(self.results[dbName][0])
                curve = self.results[dbName][1]
                total_dets = len(self.results[dbName][1][0])
                total_gts = self.DBs[dbName].getGtNum() - self.DBs[dbName].getIgnNum() 
                # print("FP: {}/{}, recall: {}/{}:{}, images: {}\n".format((1-curve[1][-1])*total_dets, total_dets, \
                #                                            curve[0][-1]*total_gts, total_gts, curve[0][-1], len(self.DBs[dbName].images)))
            
            # if not noshow:
            #     # plot curves
            #     plt.figure("recall-precision result") 
            #     plotlines = list()
            #     legends = list()
            #     for i, dbName in enumerate(self.results):
            #         #recall, precision = self.results[dbName][1]
            #         recall = self.results[dbName][1][0]
            #         precision = self.results[dbName][1][1]
            #         plotline, = plt.plot(recall, precision, color=self.colors[i])
            #         plotlines.append(plotline)
            #         legends.append("{} {:.3f}".format(dbName, self.results[dbName][0]))
            #     plt.legend(plotlines, legends, loc="lower left")
            #     plt.xlabel("recall")
            #     plt.ylabel("precision")
            #     #plt.show()
            # else:
            #     return all_mAP,(int((1-curve[1][-1])*total_dets), total_dets, \
            #                                                curve[0][-1]*total_gts, total_gts, curve[0][-1], len(self.DBs[dbName].images))
 
        else: # evaluate over each class
            self._splitClass()
            for dbName in self.splitDBs:
                mAP = 0.0
                APs = dict()
                nCls = 0
                for cls in self.splitDBs[dbName]:
                    if cls <= 0:
                        continue
                    tmpDB = self.splitDBs[dbName][cls]
                    tag = tmpDB.mapClass2Tag(cls)
                    tmpDB.compare(iouThres, matching)
                    AP, curve = tmpDB.eval_AP(metric)
                    mAP += AP
                    APs[tag] = AP
                    nCls += 1
                self.results[dbName] = [mAP/nCls, APs]
                print(dbName, "mAP = {}".format(self.results[dbName][0]))
                print(self.results[dbName][1])
    def getShowlist(self, dbName, clsID=None, option="all"):
        """
        :meth: get showlist from the evaluation results
        :param dbName: assigned database name 
        :param option: option to select showlist from all results
        :type dbName: str
        :type option: str - "all"/"fp"/"miss"/"watchlist"
        """
        assert dbName in self.DBs, "{} does not exist in self.DBs".format(dbName)
        if clsID is None:
            assert self.DBs[dbName].scorelist is not None
        else:
            assert self.splitDBs is not None and dbName in self.splitDBs
            assert self.splitDBs[dbName][clsID].scorelist is not None
        
        scorelist = self.DBs[dbName].scorelist if clsID is None else \
                        self.splitDBs[dbName][clsID].scorelist
        nrBox = len(scorelist)

        showlist = list()
        if option == "miss":
            missSet = list()
            for i, (dtbox, imageID) in enumerate(scorelist):
                if imageID in missSet: 
                    continue
                miss_num = 0
                for gtbox in self.DBs[dbName].images[imageID].gtboxes:
                    if gtbox.matched == 0 and gtbox.ign == 0:
                        miss_num += 1
                        break
                if miss_num != 0:
                    missSet.append(imageID)
                    info = DisplayInfo(imageID, "Miss", None, i, nrBox)
                    showlist.append(info)
        elif option == "fp":
            for i, (dtbox, imageID) in enumerate(scorelist):
                if dtbox.matched == 0: # false positive only
                    info = DisplayInfo(imageID, "Fp", dtbox, i, nrBox)
                    showlist.append(info)
        elif option == "all":
            for i, (dtbox, imageID) in enumerate(scorelist):
                info = DisplayInfo(imageID, "Tp" if dtbox.matched else "Fp", dtbox, i, nrBox)
                showlist.append(info)
        else:
            with open(option, "r") as f:
                watchlist = f.readlines()
            for watchterm in watchlist:
                ID, score = watchterm.strip().split()
                for i, (dtbox, imageID) in enumerate(scorelist):
                    if imageID == ID and abs(dtbox.score - float(score)) < 0.1:
                        info = DisplayInfo(imageID, "Tp" if dtbox.matched else "Fp", dtbox, i, nrBox)
                        showlist.append(info)
                        break
        return showlist


class Displayer(object):
    """
    :class: displayer to show images
    :ivar DBs: the detection results as well as groundtruth (stored in DBBase/TrainDB/EvalDB) of all algorithms, using DB.dbName as the key
    :ivar n: the number of databases
    :type DBs: dict
    :type n: int
    """
    def __init__(self, databases):
        self.DBs = dict()
        if isinstance(databases, DBBase):
            self.DBs[databases.dbName] = databases
        else:
            for db in databases:
                self.DBs[db.dbName] = db
        self.n = len(self.DBs)

    def show(self, showRef, shuffle=False, concAxis=1, maxsize=960):
        """
        :meth: show images according to the assigned sort
        :ivar showRef: the reference for showing image. It could be in 3 formats: 1) a dbName; 2) a list of DetImage.ID; and 3) a show list of DisplayInfo.
        :ivar shuffle: shuffle or not (default False)
        :ivar concAxis: concatenate axis, 1 means horizontally (default) while 0 means vertically
        :ivar maxsize: maxsize of the show image
        :type showRef: list or str
        :type shuffle: bool
        :type concAxis: int (0/1)
        :type maxsize: int 
        """
        showlist = list()
        if type(showRef) == str: # show by dbName
            db = self.DBs[showRef]
            for i, imageID in enumerate(db.images):
                info = DisplayInfo(imageID, None, None, i, len(db.images))
                showlist.append(info)
        elif type(showRef[0]) == str: # show by showlist in format 1
            for i, imageID in enumerate(showRef):
                info = DisplayInfo(imageID, None, None, i, len(showRef))
                showlist.append(info)
        else: # show by showlist in format 2
            assert type(showRef[0]) == DisplayInfo, "Unrecognized showRef format."
            showlist = showRef

        if shuffle is True:
            random.shuffle(showlist)

        i = 0
        while i < len(showlist):
            print("{}/{}: {}".format(i, len(showlist), showlist[i].sprint()))
            imgbatch = list()
            for dbName in self.DBs:
                db = self.DBs[dbName]
                imgID = showlist[i].imgID
                boldScore = showlist[i].dtbox.score if showlist[i].dtbox is not None else None
                dtShowThres = db.dtShowThres if type(db) is EvalDB else None
                img = db.images[imgID].draw(dtShowThres, boldScore)
                assert len(img.shape) == 2 or len(img.shape) == 3, 'error image shape'
                if len(img.shape) == 3:
                    h, w, d = img.shape
                elif len(img.shape) == 2:
                    h, w = img.shape
                if max(h, w) > maxsize:
                    ratio = float(maxsize)/max(h, w)
                    img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
                imgbatch.append(img)
            
            showimg = np.concatenate(imgbatch, axis=concAxis)

            '''
            cv2.imwrite( "/home/yugang/res/{}.png".format(i), showimg)
            i = i + 1
            '''
            #cv2.namedWindow(imgID, 0)
            #cv2.imshow(imgID, showimg)
            cv2.namedWindow("img", 0)
            cv2.imshow("img", showimg)
            action = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if action == 65361: #left key
                i = i - 1 if i >= 1 else 0
            elif action == 65366: #pagedown
                i = i + 20 if i < len(showlist)-21 else len(showlist)-1
            elif action == 65365: #pageup
                i = i - 20 if i >= 20 else 0
            elif action == 113: #esc
                return
            else:
                i = i + 1

