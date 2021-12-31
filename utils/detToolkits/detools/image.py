from .box import *
import os, sys
import os.path as osp
import copy
import numpy as np
import pdb
class ImageBase(object):
    """
    :class: image containing detection results and groundtruth
    :ivar str fpath: image file path
    :ivar str imgroot: image root path
    :ivar str ID: image unqiue ID
    :ivar dict dbInfo: database information
    :ivar list dtboxes: list of DetBox for all detection results
    :ivar list gtboxes: list of DetBoxGT for all detection groundtruth
    :ivar int width: width of the image 
    :ivar int height: height of the image
    :ivar int _ignNum: number of ignored groundtruth
    :ivar int _gtNum: number of total groundtruth
    :ivar int _dtNum: number of total detection results
    """
    def __init__(self, imgroot="", fpath=None, ID=None, dbInfo=None):
        self.__dict__.update(locals())
        self.dtboxes = None
        self.gtboxes = None
        self._width = None
        self._height = None
        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def __str__(self):
        return str(self.dumpOdf("all"))

    @property
    def width(self):
        """
        :return: the width of the image
        """
        if self._width is None:
            return self.getShape[0]
        return self._width

    @property
    def height(self):
        """
        :return: the height of the image
        """
        if self._height is None:
            return self.getShape[1]
        return self._height

    @property
    def getShape(self):
        """
        :return: the [width, height] of the image
        """
        if self._width is None or self._height is None:
            img = None
            imgpath = osp.join(self.imgroot, self.fpath)
            img = cv2.imread(imgpath)
            self._width, self._height = img.shape[1], img.shape[0]
        return self._width, self._height

    def getDistribution(self, gtORdt, *args):
        """
        :meth: get the distribution of the assigned variable in the image
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt")
        :param \*args: the rule to get the variable from each bounding box, args[0] is the function while args[1:] are the other params if needed
        :type gtORdt: str - "gt"/"dt"        
        :return: a list of the assigned variables
        :example: getDistribution("dt", lambda box:box.h) will count the distribution of the heights of detection results 
        """
        if gtORdt == "gt":
            boxes = self.gtboxes
        elif gtORdt == "dt":
            boxes = self.dtboxes
        else:
            print("Please choose gt or dt.")
            return

        distr = list()
        func = args[0]
        for box in boxes:
            val = func(box, *args[1:])
            if val is not None:
                distr.append(val)
        return distr

    def getGtNum(self):
        """
        :meth: get the number of groundtruth boxes in the image, as self._gtNum
        :return: the total groundtruth number
        """
        if self._gtNum is None:
            self._gtNum = len(self.gtboxes) if self.gtboxes is not None else 0
        return self._gtNum

    def getDtNum(self):
        """
        :meth: get the number of detection result boxes in the image, as self._dtNum
        :return: the total detection result number
        """
        if self._dtNum is None:
            self._dtNum = len(self.dtboxes) if self.dtboxes is not None else 0
        return self._dtNum

    def getIgnNum(self):
        """
        :meth: get the number of ignored groundtruth boxes in the image, as self._ignNum
        :return: the ignored groundtruth number
        """
        if self._ignNum is None:
            self._ignNum = 0
            if self.gtboxes is not None:
                for gtbox in self.gtboxes:
                    self._ignNum += gtbox.ign
        return self._ignNum

    def doNMS(self, *args):
        """
        :meth: merge the bounding boxes of detection results
        :param \*args: the rule for NMS merging given two boxes, args[0] is the function (return 1 for merging otherwise 0) while args[1:] are the other params if needed
        :example: doNMS(lambda a,b,thr:a.iom(b)>thr, 0.65) will do NMS merging for all bounding boxes pair with iomin>0.65
        """
        if self.dtboxes is not None:
            dtboxesNMS = list()
            self.dtboxes.sort(key=lambda x: x.score, reverse=True)
            dtNum = len(self.dtboxes)
            usedflags = [0] * dtNum
            func = args[0]
        
            for i in range(dtNum):
                if usedflags[i] == 1: continue
                for j in range(i+1, dtNum):
                    if usedflags[j] == 1: continue
                    if func(self.dtboxes[i], self.dtboxes[j], *args[1:]): usedflags[j] = 1
                dtboxesNMS.append(self.dtboxes[i])
            self.dtboxes = dtboxesNMS

    def filterBoxes(self, gtORdt, *args):
        """
        :meth: filter the bounding boxes of detection results or groundtruth
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt")
        :param \*args: the rule for filtering a bounding box, args[0] is the function (return 0 for filtering otherwise 1) while args[1:] are the other params if needed
        :type gtORdt: str - "gt"/"dt"        
        :example: filterBoxes("dt", lambda a,thr:a.score>thr, 110) will filter all detection boxes whose scores lower than 110
        """
        if gtORdt == "gt":
            boxes = self.gtboxes
        elif gtORdt == "dt":
            boxes = self.dtboxes
        else:
            print("Please choose gt or dt.")
            return

        if boxes is not None:
            boxesFiltered = list()
            func = args[0]
            for box in boxes:
                if func(box, *args[1:]): boxesFiltered.append(box)
            if gtORdt == "gt":
                self.gtboxes = boxesFiltered
            else:
                self.dtboxes = boxesFiltered
    def clip_boader(self,odf,name):
        assert name in odf 
        x,y,w,h = odf[name]
        x = np.max([0,x])
        y = np.max([0,y])
        w = np.min([x+w,self._width]) - x
        h = np.min([y+h,self._height]) - y
        odf[name] = [x,y,w,h]
        return odf
    def parseOdf(self, odf, gtORdt="all"):
        """
        :meth: read the object from a dict
        :param odf: a dict with "fpath", "ID", "dbInfo", "gtboxes" and "dtboxes" keys, e.g., odf = {"fpath": "test.jpg", "ID": "Test", "dbInfo": "TestDB", dtboxes: [{"box": [5,5,20,50], "score": 1.0}]}
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt") or both ("all", default)
        :type odf: dict
        :type gtORdt: str - "gt"/"dt"/"all"
        """
        if gtORdt == "gt":
            gtflag, dtflag = True, False
        elif gtORdt == "dt":
            gtflag, dtflag = False, True
        elif gtORdt == "all":
            gtflag, dtflag = True, True
        else:
            print("Please choose gt or dt or all.")
            return

        if self.fpath is None and "fpath" in odf:
            self.fpath = odf["fpath"]
        if self.ID is None and "ID" in odf:
            self.ID = odf["ID"]
        if self.dbInfo is None and "dbInfo" in odf:
            self.dbInfo = odf["dbInfo"]
        if self._width is None and "width" in odf:
            self._width = odf["width"]
        if self._height is None and "height" in odf:
            self._height = odf["height"]

        
        if gtflag and "gtboxes" in odf and odf["gtboxes"] is not None:
            self.gtNum = len(odf["gtboxes"])
            self.gtboxes = list()
            for gtbox in odf["gtboxes"]:
                gt = DetBoxGT()
                gtbox = self.clip_boader(gtbox,'fbox')
                gt.parseOdfbyname(gtbox,'fbox')
                
                self.gtboxes.append(gt)
        if gtflag and 'gt_boxes' in odf and odf['gt_boxes'] is not None:
            self.gtNum = len(odf["gt_boxes"])
            self.gtboxes = list()
            for gtbox in odf["gt_boxes"]:
                gt = DetBoxGT()
                gtbox = self.clip_boader(gtbox,'fbox')
                gt.parseOdfbyname(gtbox,'fbox')
                self.gtboxes.append(gt)
        if dtflag and "dtboxes" in odf and odf["dtboxes"] is not None:
            self.dtboxes = list()
            for dtbox in odf["dtboxes"]:
                dt = DetBox()
                dtbox = self.clip_boader(dtbox,'box')
                dt.parseOdf(dtbox)
                self.dtboxes.append(dt)
        if dtflag and 'dt_boxes' in odf and odf['dt_boxes'] is not None:
            self.dtboxes = list()
            for dtbox in odf["dt_boxes"]:
                dt = DetBox()
                dtbox = self.clip_boader(dtbox,'box')
                dt.parseOdf(dtbox)
                self.dtboxes.append(dt)
    def dumpOdf(self, gtORdt=None):
        """
        :meth: dump the object into a dict
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt") or both ("all", default)
        :type gtORdt: str - "gt"/"dt"/"all"
        """
        if gtORdt == "gt":
            gtflag, dtflag = True, False
        elif gtORdt == "dt":
            gtflag, dtflag = False, True
        elif gtORdt == "all":
            gtflag, dtflag = True, True
        else:
            print("Please choose gt or dt or all.")
            return

        odf = dict()
        if self.fpath is not None:
            odf["fpath"] = self.fpath
        if self.ID is not None: 
            odf["ID"] = self.ID
        if self.dbInfo is not None:
            odf["dbInfo"] = self.dbInfo
        if self.width is not None:
            odf["width"] = self.width
        if self.height is not None:
            odf["height"] = self.height

        if gtflag and self.gtboxes is not None:
            odf["gtboxes"] = list()
            for gtbox in self.gtboxes:
                odf["gtboxes"].append(gtbox.dumpOdf())
        if dtflag and self.dtboxes is not None:
            odf["dtboxes"] = list()
            for dtbox in self.dtboxes:
                odf["dtboxes"].append(dtbox.dumpOdf())
        return odf

    def draw(self, thres=None, boldScore=None):
        """
        :meth: draw all bounding boxes in the image 
        :param thres: threshold for drawing detection results
        :param boldScore: the detection results whose scores are equal to boldScore will be drawn in bold
        :type thres: float
        :type boldScore: float
        :return: the result image
        """
        img = self.imread()
        if self.dtboxes is not None:
            for bbox in self.dtboxes:
                if thres is not None and bbox.score < thres:
                    continue
                bold = boldScore is not None and abs(float(bbox.score) - float(boldScore)) < 0.01
                bbox.draw(img, bold=bold) 
        if self.gtboxes is not None:
            for bbox in self.gtboxes:
                bbox.draw(img)
        return img

    def imread(self):
        """
        :meth: read the image from imgroot + fpath. If TrainImage.flipped is True, the returned image will be flipped
        :return: the image mat
        """
        
        imgpath = os.path.join(self.imgroot, self.fpath)
        img = cv2.imread(imgpath)

        assert img is not None, 'image {} do not exist'.format(self.fpath)

        self._width, self._height = img.shape[1], img.shape[0]

        if "flipped" in self.__dict__ and self.flipped is True:
            img = cv2.flip(img, 1)
        return img 
        
    def splitClass(self, tag2class):
        """
        :meth: split the image into several images, each containing bboxes of one class (not one tag)
        :param tag2class: a function or dict mapping a tag (str) to a class (int)
        :type tag2class: function or dict
        :return: a dict of ImageBases, with class ID (int) as the key
        """
        def _clone():
            image = self.__class__(self.imgroot, self.fpath, self.ID, self.dbInfo)
            image._width, image._height = self._width, self._height
            if "flipped" in self.__dict__:
                image.flipped = self.flipped
            return image

        images = dict()
        if self.dtboxes is not None:
            for bbox in self.dtboxes:
                cls = tag2class(bbox.tag)
                if cls not in images:
                    images[cls] = _clone()
                if images[cls].dtboxes is None:
                    images[cls].dtboxes = list()
                images[cls].dtboxes.append(bbox)

        if self.gtboxes is not None:
            for bbox in self.gtboxes:
                cls = tag2class(bbox.tag)
                if cls not in images:
                    images[cls] = _clone()
                if images[cls].gtboxes is None:
                    images[cls].gtboxes = list()
                images[cls].gtboxes.append(bbox)
        return images


class TrainImage(ImageBase):
    """
    :class: image for training, inherited from ImageBase
    :ivar bool flipped: the image has been flipped or not, default False
    """
    def __init__(self, imgroot="", fpath=None, ID=None, dbInfo=None):
        super(TrainImage, self).__init__(imgroot, fpath, ID, dbInfo)
        self.flipped = False

    def doFlip(self):
        """
        :meth: horizontally flip the bounding boxes of the image
        :return: the flipped TrainImage and the flipped image numpy array
        """
        trainImage = copy.deepcopy(self)
        trainImage.flipped = not trainImage.flipped
        img = trainImage.imread()

        if trainImage.dtboxes is not None:
            for bbox in trainImage.dtboxes:
                bbox.x = trainImage.width - 1 - bbox.x1
        if trainImage.gtboxes is not None:
            for bbox in trainImage.gtboxes:
                bbox.x = trainImage.width - 1 - bbox.x1
        return trainImage, img


class EvalImage(ImageBase):
    """
    :class: image for evaluation by comparing detection results with groundtruths, inherited from ImageBase
    """
    def _resetMatches(self):
        """
        :meth: reset all the dtbox.match and gtbox.match to 0
        """
        if self.dtboxes is not None:
            for bbox in self.dtboxes:
                bbox.matched = 0
        
        if self.gtboxes is not None:
            for bbox in self.gtboxes:
                bbox.matched = 0

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        self._resetMatches()

        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt.matched == 1:
                    continue
                if gt.ign == 0:
                    overlap = dt.iou(gt)
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    
                    if maxpos >= 0:
                        break
                    else:
                        overlap = dt.ioa(gt)
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        self._resetMatches()

        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist