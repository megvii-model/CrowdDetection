import json
import numpy as np
import copy
import random
from .box import *
from .image import *
import pdb
CALTECH_CLASSES = { "__background__": 0,
                    "person": 1 }
PERSON_CLASSES = { "__background__": 0,
                  "person": 1 }

PERSON_FPv1_CLASSES = { "__background__": 0,
                  "person": 1,
                       "traffic light": 2}

PERSONC3_CLASSES = { "__background__": 0,
                    "person": 1, "umbrella": 2, "rider": 3}

PERSONC3TOC1_CLASSES = { "__background__": 0,
                       "person": 1, "umbrella": 1, "rider": 1}

HEAD_CLASSES = { "__background__": 0,
                    "head": 1 }

FACE_CLASSES = { "__background__": 0,
                    "Face": 1 }


#deprecated
VEHICLE_CLASSES = { "__background__": 0,
                   "Car": 1, 'Jeep': 2, 'Macrobus': 3, 'Minibus': 4, 'Bus': 5, 'SmallTruck': 6, 'Truck': 7, 'Bicycle': 8, 'electronmobile': 9, 'motobike': 10, 'tricycle': 11, 'Others': 12}


#VEHICLEC5_CLASSES = { '__background__': 0,
#                             'car': 1, 'bus': 2, 'truck': 3, 'bicycle': 4, 'motorcycle': 5 }

VEHICLEC5_CLASSES = { '__background__': 0,
                                          'car': 1, 'suv': 1, 'microbus': 1,
                                          'midibus': 2, 'bus': 2, 
                                          'pickup': 3, 'truck': 3, 
                                          'bicycle': 4, 
                      'motorcycle': 5, 'electrocycle': 5, 'tricycle': 5 }

HVC6_CLASSES = { '__background__': 0,
                'person': 1, 'car': 2, 'suv': 2, 'microbus': 2,
                                          'midibus': 3, 'bus': 3, 
                                          'pickup': 4, 'truck': 4, 
                                          'bicycle': 5, 
                      'motorcycle': 6, 'electrocycle': 6, 'tricycle': 6 }

HVC7_CLASSES = { '__background__': 0,
                'person': 1, 'car': 2, 'suv': 7, 'microbus': 2,
                                          'midibus': 3, 'bus': 3, 
                                          'pickup': 4, 'truck': 4, 
                                          'bicycle': 5, 
                      'motorcycle': 6, 'electrocycle': 6, 'tricycle': 6 }

HVC8_CLASSES = { '__background__': 0,
                'person': 1, 'car': 2, 'suv': 2, 'microbus': 2,
                                          'midibus': 3, 'bus': 3, 
                                          'pickup': 4, 'truck': 4, 
                                          'bicycle': 5, 
                      'motorcycle': 6, 'electrocycle': 6, 'tricycle': 6 ,
                    'rider': 7, 'umbrella':8
               }

SHUGUANG_CLASSES = { '__background__': 0,
                'person': 1, 'car': 2, 'suv': 3, 'microbus': 4,
                                          'midibus': 5, 'bus': 5, 
                                          'pickup': 6, 'truck': 7, 
                                          'bicycle': 8, 
                      'motorcycle': 8, 'electrocycle': 8, 'tricycle': 8 ,
               }

#shuguangv2
SGV2_CLASSES = { '__background__': 0,
                'person': 1, 'car': 2, 'suv': 3, 'microbus': 4,
                                          'midibus': 5, 'bus': 5, 
                                          'pickup': 6, 'truck': 7, 
                                          'bicycle': 8, 
                      'motorcycle': 8, 'electrocycle': 8, 'tricycle': 8 ,
                       'rider': 9,
               }

VEHICLEC3_CLASSES = { '__background__': 0,
                                          'car': 1, 'suv': 1, 'microbus': 1,
                                          'midibus': 2, 'bus': 2, 
                                          'pickup': 3, 'truck': 3}


VEHICLEC10_CLASSES = { '__background__': 0,
                                          'car': 1, 'suv': 2, 'microbus': 3,
                                          'midibus': 4, 'bus': 5, 
                                          'pickup': 6, 'truck': 7, 
                                          'bicycle': 8, 
                                          'motorcycle': 9, 'electrocycle': 9, 'tricycle': 10 }

Car4W_CLASSES = { '__background__': 0, 
                                          'car': 1, 'suv': 1, 'microbus': 1,
                                          'midibus': 1, 'bus': 1, 
                                          'pickup': 1, 'truck': 1}

KITTI_CLASSES = { "__background__": 0, 
                  "Person_sitting": -1, "DontCare": -1, "Misc": -1,
                  "Pedestrian": 1, "Cyclist": 2 }

VOC_CLASSES = { "__background__": 0, 
                "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                "motorbike": 14, "person": 15, "pottedplant": 16,
                "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20 }

COCO_CLASSES = { "__background__": 0, "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10, "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14, "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20, "elephant": 21, "bear": 22, "zebra": 23, "giraffe": 24, "backpack": 25, "umbrella": 26, "handbag": 27, "tie": 28, "suitcase": 29, "frisbee": 30, "skis": 31, "snowboard": 32, "sports ball": 33, "kite": 34, "baseball bat": 35, "baseball glove": 36, "skateboard": 37, "surfboard": 38, "tennis racket": 39, "bottle": 40, "wine glass": 41, "cup": 42, "fork": 43, "knife": 44, "spoon": 45, "bowl": 46, "banana": 47, "apple": 48, "sandwich": 49, "orange": 50, "broccoli": 51, "carrot": 52, "hot dog": 53, "pizza": 54, "donut": 55, "cake": 56, "chair": 57, "couch": 58, "potted plant": 59, "bed": 60, "dining table": 61, "toilet": 62, "tv": 63, "laptop": 64, "mouse": 65, "remote": 66, "keyboard": 67, "cell phone": 68, "microwave": 69, "oven": 70, "toaster": 71, "sink": 72, "refrigerator": 73, "book": 74, "clock": 75, "vase": 76, "scissors": 77, "teddy bear": 78, "hair drier": 79, "toothbrush": 80, 
}

DETRAC_CLASSES = {'__background__': 0, 'car': 1, 'bus':2 , 'truck':3, }

# DBBase
class DBBase(object):
    """ :class: database containing a set of Images for evaluation
    :ivar str dbName: database name, the unique ID
    :ivar dict images: dict of Images (EvalImage/TrainImage/ImageBase), using imageID as the key
    :ivar dict _classes: a mapping from tags to class ID
    :ivar dict _classesInv: a mapping from class ID to tags
    :ivar int _ignNum: number of ignored groundtruth in the whole database
    :ivar int _gtNum: number of total groundtruth in the whole database
    :ivar int _dtNum: number of total detection results in the whole database
    """
    def __init__(self, dbName, imgroot="", gtpath=None, dtpath=None):
        self.dbName = dbName
        self.imgroot = imgroot
        self.images = None
        self._classes = None
        self._classesInv = None
        self._ignNum = None
        self._gtNum = None
        self._dtNum = None
        if gtpath is not None:
            self.loadOdf(gtpath, "gt")
        if dtpath is not None:
            self.loadOdf(dtpath, "dt")

        # automatically set classes by dbName
        if 'voc' in dbName.lower():
            self.setClasses(VOC_CLASSES)
        elif 'human' in dbName.lower():
            self.setClasses(PERSON_CLASSES)
        elif 'caltech' in dbName.lower():
            self.setClasses(CALTECH_CLASSES)
        elif 'kitti' in dbName.lower():
            self.setClasses(KITTI_CLASSES)
        elif 'coco' in dbName.lower():
            self.setClasses(COCO_CLASSES)
        elif 'detrac' in dbName.lower():
            self.setClasses(DETRAC_CLASSES)
        elif 'hvc6' in dbName.lower():
            self.setClasses(HVC6_CLASSES)
        elif 'hvc7' in dbName.lower():
            self.setClasses(HVC7_CLASSES)
        elif 'hvc8' in dbName.lower():
            self.setClasses(HVC8_CLASSES)
        elif 'personc3' in dbName.lower():
            self.setClasses(PERSONC3_CLASSES)
        elif 'person_fpv1' in dbName.lower():
            self.setClasses(PERSON_FPv1_CLASSES)
        elif 'shuguang' in dbName.lower():
            self.setClasses(SHUGUANG_CLASSES)
        elif 'sgv2' in dbName.lower():
            self.setClasses(SGV2_CLASSES)
        elif 'personc3toc1' in dbName.lower():
            self.setClasses(PERSONC3TOC1_CLASSES)
        elif 'car4w' in dbName.lower():
            self.setClasses(Car4W_CLASSES)
        elif 'vehiclec5' in dbName.lower():
            self.setClasses(VEHICLEC5_CLASSES)
        elif 'vehiclec3' in dbName.lower():
            self.setClasses(VEHICLEC3_CLASSES)
        else:
            self.setClasses(COCO_CLASSES)
            #pass

    def setClasses(self, classes):
        """
        :meth: set self._classes and self._classesInv 
        :param classes: a mapping from tag to class ID, e.g. {"person": 1, "background":0}
        :type classes: dict
        """
        self._classes = classes
        self._classesInv = dict()
        for tag in classes:
            cls = classes[tag]
            if cls not in self._classesInv:
                if cls < 0:
                    self._classesInv[cls] = 'Ignore+{}'.format(tag)
                else:
                    self._classesInv[cls] = tag
            else:
                self._classesInv[cls] += '+{}'.format(tag)

    def mapTag2Class(self, tag):
        """
        :meth: map a tag to class ID
        :param tag: tag
        :type tag: str
        :return: class ID if tag existed in self._classes else 0
        """
        assert self._classes is not None, "self.setClasses() should be set first!"
        assert tag in self._classes, "tag {} not found!".format(tag)
        return self._classes[tag]

    def mapClass2Tag(self, classID):
        """
        :meth: map a class ID to tag
        :param class: class ID
        :type class: int
        :return: tag if class ID existed in self._classesInv else 0
        """
        assert self._classesInv is not None, "self.setClasses() should be set first!"
        assert classID in self._classesInv, "classID {} not found!".format(classID)
        return self._classesInv[classID]

    def loadOdf(self, fpath, gtORdt="all"):
        """
        :meth: read the object from a file
        :param fpath: the odf file path, in which each line records a dict as ImageBase.parseOdf() input
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt") or both ("all", default)
        :type fpath: str
        :type gtORdt: str - "gt"/"dt"/"all"
        """
        assert os.path.isfile(fpath), fpath + " does not exist!"
        
        # key_name = 'fbox'
        if self.images is None:
            self.images = dict()
        with open(fpath, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            odf = json.loads(line)
            if odf["ID"] not in self.images:
                self.images[odf["ID"]] = self._newImage()
            self.images[odf["ID"]].parseOdf(odf, gtORdt)
            # self.images[odf["ID"]].parseOdfbyname(odf, gtORdt,key_name)

    def saveOdf(self, fpath, gtORdt="all"):
        """
        :meth: write the object into a file
        :param fpath: the odf file path, in which each line records a dict as ImageBase.dumpOdf()
output
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt") or both ("all", default)
        :type fpath: str
        :type gtORdt: str - "gt"/"dt"/"all"
        """
        if self.images is not None:
            with open(fpath, "w") as f:
                for ID in self.images:
                    json.dump(self.images[ID].dumpOdf(gtORdt), f)
                    f.write("\n")
    
    def _newImage(self):
        """
        :meth: construct a new Image object
        :return: a new ImageBase(imgroot=self.imgroot) for DBBase class
        """
        return ImageBase(imgroot=self.imgroot)

    def getDistribution(self, gtORdt, *args):
        """
        :meth: get the distribution of the assigned variable in the whole database
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt")
        :param \*args: the rule to get the variable from each bounding box, args[0] is the function while args[1:] are the other params if needed
        :type gtORdt: str - "gt"/"dt"        
        :return: a list of the assigned variables
        :seealso: ImageBase.getDistribution()
        """
        distr = list()
        for ID in self.images:
            distr.extend(self.images[ID].getDistribution(gtORdt, *args))
        return distr

    def getGtNum(self):
        """
        :meth: get the number of groundtruth boxes in the database, as self._gtNum
        :return: the total goundtruth number
        :seealso: ImageBase.getGtNum()
        """
        if self._gtNum is None:
            self._gtNum = 0
            if self.images is not None:
                for ID in self.images:
                    self._gtNum += self.images[ID].getGtNum()
        return self._gtNum

    def getDtNum(self):
        """
        :meth: get the number of detection result boxes in the database, as self._dtNum
        :return: the total detection result number
        :seealso: ImageBase.getDtNum()
        """
        if self._dtNum is None:
            self._dtNum = 0
            if self.images is not None:
                for ID in self.images:
                    self._dtNum += self.images[ID].getDtNum()
        return self._dtNum

    def getIgnNum(self):
        """
        :meth: get the number of ignored groundtruth boxes in the database, as self._ignNum
        :return: the ignored goundtruth number
        :seealso: ImageBase.getIgnNum()
        """
        if self._ignNum is None:
            self._ignNum = 0
            if self.images is not None:
                for ID in self.images:
                    self._ignNum += self.images[ID].getIgnNum()
        return self._ignNum

    def getImageNum(self):
        """
        :meth: count the number of images
        :return: the number of images
        """
        return len(self.images)

    def doNMS(self, *args):
        """
        :meth: merge the bounding boxes of detection results in the whole database
        :param \*args: the rule for NMS merging given two boxes, args[0] is the function (return 1 for merging otherwise 0) while args[1:] are the other params if needed
        :seealso: ImageBase.doNMS()
        """
        if self.images is not None:
            for ID in self.images:
                self.images[ID].doNMS(*args)

    def filterBoxes(self, gtORdt, *args):
        """
        :meth: filter the bounding boxes of detection results or groundtruth in the whole database
        :param gtORdt: choose detection results ("dt") or groundtruth ("gt")
        :param \*args: the rule for filtering a bounding box, args[0] is the function (return 0 for filtering otherwise 1) while args[1:] are the other params if needed
        :type gtORdt: str - "gt"/"dt"        
        :seealso: ImageBase.filterBoxes()
        """
        if self.images is not None:
            for ID in self.images:
                self.images[ID].filterBoxes(gtORdt, *args)

    def splitClass(self):
        """
        :meth: split the dataset into several subsets, each containing bbox of one class (not one tag)
        :seealso: ImageBase.splitClass
        :return: a dict of DBBases, with class ID (int) as the key
        """
        def _clone():
            db = self.__class__(self.dbName, self.imgroot)
            db._classes, db._classesInv = self._classes, self._classesInv
            return db

        dbs = dict()
        if self.images is not None:
            for ID in self.images:
                splitImages = self.images[ID].splitClass(self.mapTag2Class)
                for cls in splitImages:
                    if cls not in dbs:
                        dbs[cls] = _clone()
                        dbs[cls].images = dict()
                    dbs[cls].images[ID] = splitImages[cls]
        return dbs


# TrainDB
class TrainDB(DBBase):
    """
    :class: database containing a set of TrainImages for training
    :ivar int _currentIdx: current minibatch idx
    :ivar list _imgseq: list of TrainImages for generating minibatch
    :ivar int _randgen: random generator
    """
    def __init__(self, dbName, imgroot="", gtpath=None, dtpath=None):
        super(TrainDB, self).__init__(dbName, imgroot, gtpath, dtpath)
        self._currentIdx = None
        self._imgseq = None
        self._randgen = None

    def _newImage(self):
        """
        :meth: construct a new Image object
        :return: a new TrainImage(imgroot=self.imgroot) for TrainDB class
        """
        return TrainImage(imgroot=self.imgroot)

    def filterImages(self):
        """
        :meth: filter images without any groundtruth boxes 
        """
        if self.images is not None:
            delIDs = list()
            for ID in self.images:
                if self.images[ID].getGtNum() == 0:
                    delIDs.append(ID)
            for ID in delIDs:
                del self.images[ID]

    def setRandomSeed(self, seed=23333):
        """
        :meth: set random seed as self._randgen
        :param seed: random seed
        :type seed: int
        """
        self._randgen = random.Random(seed)

    def _shuffleImages(self):
        """
        :meth: random shuffle images from self.images (dict) into self._imgseq (list), reset self._currentIdx = 0
        """
        self._imgseq = list()
        if self.images is not None:
            for ID in self.images:
                self._imgseq.append(self.images[ID])

        if self._randgen is None:
            self.setRandomSeed()
        self._randgen.shuffle(self._imgseq)
        self._currentIdx = 0

    def getOneBatch(self, sizeRange=None, sizeContinuous=False, maxsize=2048, doFlip=True):
        """
        :meth: get one minibatch with batchsize = 1
        :param sizeRange: the range of min length of image side, in the format of tuple. If sizeContinuous is True, only 2D-tuple is allowed
        :param sizeContinuous: if True, a continuous varible falling in sizeRange will be thrown out randomly; otherwise, a discrete varible of sizeRange will be choosen randomly (default)
        :param maxsize: the upbound of max image dimension
        :type sizeRange: tuple
        :type sizeContinuous: bool
        :type maxsize: int
        :return: a dict containing imgBlob and boxBlob, in the format of {"img": imgBlob, "gt_boxes": boxBlob, "imgID": I0000, "imscale": 1.2}
        """
        if sizeContinuous is True:
            assert len(sizeRange) == 2 and sizeRange[0] < sizeRange[1]

        if self._currentIdx is None or self._currentIdx >= self.getImageNum():
            self._shuffleImages()

        doFlip = doFlip is True and self._randgen.randint(0,100) % 2 == 0
        if doFlip:
            image, imgBlob = self._imgseq[self._currentIdx].doFlip()
        else:
            image = self._imgseq[self._currentIdx]
            imgBlob = image.imread()

        width, height = image.width, image.height
        minDim, maxDim = min(width, height), max(width, height)
        maxscale = maxsize / float(maxDim)

        imscale = 1.0
        if sizeContinuous is False:
            imsize = self._randgen.choice(sizeRange)
            imscale = imsize / float(minDim)
        else:
            imsize = self._randgen.randrange(sizeRange[0], sizeRange[1])
            imscale = imsize / float(minDim)

        imscale = min(maxscale, imscale)
        imgBlob = cv2.resize(imgBlob, (int(np.floor(width*imscale)), int(np.floor(height*imscale))))

        boxBlob = np.zeros((image.getGtNum(), 5), dtype=np.float)
        if image.getGtNum() > 0:
            for i, gtbox in enumerate(image.gtboxes):
                boxBlob[i, 0] = gtbox.x * imscale
                boxBlob[i, 1] = gtbox.y * imscale
                boxBlob[i, 2] = gtbox.x1 * imscale
                boxBlob[i, 3] = gtbox.y1 * imscale
                boxBlob[i, 4] = self.mapTag2Class(gtbox.tag) if gtbox.ign == 0 else -1

        self._currentIdx += 1
        return { "img": imgBlob, "gt_boxes": boxBlob, \
                "imgID": image.ID, "imscale": imscale, "flipped": doFlip }


# EvalDB
class EvalDB(DBBase):
    """
    :class: database containing a set of EvalImages for evaluation
    :ivar float dtShowThres: threshold for drawing detection results
    :ivar list scorelist: the comparison result for one class
    :ivar list CALTECH_MRREF_2: anchor points (from 10^-2 to 1) for calculating log-average miss rate, as in P.Dollar's paper
    :ivar list CALTECH_MRREF_4: anchor points (from 10^-4 to 1) for calculating log-average miss rate, as in S.Zhang's paper
    """
    def __init__(self, dbName, imgroot="", gtpath=None, dtpath=None, showthres=0.0):
        super(EvalDB, self).__init__(dbName, imgroot, gtpath, dtpath)
        self.dtShowThres = showthres
        self.scorelist = None

    def _newImage(self):
        """
        :meth: construct a new Image object
        :return: a new EvalImage(imgroot=self.imgroot) for EvalDB class
        """
        return EvalImage(imgroot=self.imgroot)

    def filter(self, score_thresh = 0.01):
        assert score_thresh>=0
        if self.scorelist is None:
            self.compare()
        scorelist = list(filter(lambda rb:rb[0].score>=score_thresh,self.scorelist))
        self.scorelist = scorelist
    def compare(self, thres=0.5, matching=None):
        """
        :meth: match the detection results with the groundtruth in the whole database
        :param thres: iou threshold, default 0.5
        :param matching: matching strategy, default None (Caltech matching strategy)
        :type thres: float
        :type matching: str - None/"VOC"
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score, as self.scorelist or self.allScorelists
        """
        assert matching is None or matching == "VOC", matching
        imageNum = self.getImageNum()
        scorelist = list()
        for ID in self.images:

            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)

        scorelist.sort(key=lambda x: x[0].score, reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2"):
        """
        :meth: evaluate by Caltech-style log-average miss rate
        :param ref: anchor points for calculating log-average miss rate, default "CALTECH_-2"
        :type ref: str - "CALTECH_-2"/"CALTECH_-4"
        :return: 1, log-average miss rate; 2, fppi-miss curve
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst)-1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]
            
        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[0].matched == 1:
                tp += 1.0
            else:
                fp += 1.0

            fn = (self.getGtNum() - self.getIgnNum()) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            missrate = 1.0 - recall
            fppi = fp / self.getImageNum()
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean()) # average miss rate
        return MR, (fppiX, fppiY)       

    def eval_AP(self, metric=None):
        """
        :meth: evaluate by average precision
        :param metric: metric for calculating AP, default None
        :type metric: str - None/"VOC"
        :return: 1, average precision; 2, recall-precision curve
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area
        
        # calculate voc ap score
        def _calculate_voc(recall, precision):
            assert len(recall) == len(precision)
            recall, precision = np.array(recall), np.array(precision)
            area = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                area += p / 11.
            return area

        assert metric is None or metric == "VOC", metric

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[0].matched == 1:
                tp += 1.0
            else:
                fp += 1.0

            fn = (self.getGtNum() - self.getIgnNum()) - tp
            try:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                rpX.append(recall)
                rpY.append(precision)
            except:
                pass

        if metric == "VOC":
            AP = _calculate_voc(rpX, rpY)
        else:
            AP = _calculate_map(rpX, rpY)
        return AP, (rpX, rpY)       


    def eval_AP_detail(self, metric=None):
        """
        :meth: evaluate by average precision
        :param metric: metric for calculating AP, default None
        :type metric: str - None/"VOC"
        :return: 1, average precision; 2, recall-precision curve
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area
        

        # calculate voc ap score
        def _calculate_voc(recall, precision):
            assert len(recall) == len(precision)
            recall, precision = np.array(recall), np.array(precision)
            area = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                area += p / 11.
            return area

        assert metric is None or metric == "VOC", metric

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        total_det = len(self.scorelist)
        total_gt = self.getGtNum() - self.getIgnNum()
        total_images = self.getImageNum()
        
        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[0].matched == 1:
                tp += 1.0
            else:
                fp += 1.0

            fn = (self.getGtNum() - self.getIgnNum()) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0].score)
            fppi.append(fp/total_images)
            

        if metric == "VOC":
            AP = _calculate_voc(rpX, rpY)
        else:
            AP = _calculate_map(rpX, rpY)
        #return AP, (thr, fpn, recalln, fppi, rpX, rpY)       
        return AP, (rpX, rpY, thr, fpn, recalln, fppi)   

