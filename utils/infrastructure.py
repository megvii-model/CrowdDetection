#coding:utf-8
import os, sys
from box import *
from matching import maxWeightMatching
from scipy.optimize import linear_sum_assignment
import numpy as np
import pdb
def minimumWeightMatching(costSet : np.ndarray) -> list:
    '''
    Computes a minimum-weight matching in a bipartite graph
    (A union B, E).

    costSet:
    An (m x n)-matrix of real values, where costSet[i, j]
    is the cost of matching the i:th vertex in A to the j:th 
    vertex of B. A value of numpy.inf is allowed, and is 
    interpreted as missing the (i, j)-edge.

    returns:
    A minimum-weight matching given as a list of pairs (i, j), 
    denoting that the i:th vertex of A be paired with the j:th 
    vertex of B.
    '''

    m, n = costSet.shape
    nMax = max(m, n)

    # Since the choice of infinity blocks later choices for that index, 
    # it is important that the cost matrix is square, so there
    # is enough space to shift the choices for infinity to the unused 
    # part of the cost-matrix.
    costSet_ = np.full((nMax, nMax), np.inf)

    mask = costSet < 0
    costSet_[:m, :n][mask] = costSet[mask]
    assert costSet_.shape[0] == costSet_.shape[1]
    
    # We allow a cost to be infinity. Since scipy does not
    # support this, we use a workaround. We represent infinity 
    # by M = 2 * maximum cost + 1. The point is to choose a distinct 
    # value, greater than any other cost, so that choosing an 
    # infinity-pair is the last resort. The 2 times is for large
    # values for which x + 1 == x in floating point. The plus 1
    # is for zero, for which 2 x == x.
    try:
        practicalInfinity = 2 * costSet[costSet < np.inf].max() + 10
    except ValueError:
        # This is thrown when the indexing set is empty;
        # then all elements are infinities.
        practicalInfinity = 1

    # Replace infinitites with our representation.
    costSet_[costSet_ == np.inf] = practicalInfinity

    # Find a pairing of minimum total cost between matching second-level contours.
    iSet, jSet = linear_sum_assignment(costSet_)
    assert len(iSet) == len(jSet)

    # Return only pairs with finite cost.
    indices = [(iSet[k], jSet[k]) 
        for k in range(len(iSet)) 
        if costSet_[iSet[k], jSet[k]] != practicalInfinity]

    return indices

def compute_lap(dtboxes, gtboxes, thr):

    eps = 1e-7
    n, k = dtboxes.shape[0], gtboxes.shape[0]
    if k + n < 2:
        m, n = np.array([]), np.array([])
        return m, n
        
    overlaps = compute_iou_matrix(dtboxes, gtboxes)

    if n < 2:
        cols = np.argmax(overlaps, axis = 1)
        rows = np.array([0])
        m, n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n

    if k < 2:
        
        rows = np.argmax(overlaps, axis = 0)
        cols = np.array([0])
        m,n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n
            
    ious = overlaps * (overlaps >= thr)
        
    matches = minimumWeightMatching(-ious)
    m, n = np.array([i for i, _ in matches]).astype(np.int32), np.array([i for _, i in matches]).astype(np.int32)
    indice = np.where(overlaps[m, n] < thr)[0]

    if indice.size >= m.size:
        m, n = np.array([]), np.array([])
    else:
        index = np.array(list(set(np.arange(m.size)) - set(indice))).astype(np.int)
        m, n = m[index], n[index]
    
    return m, n

def compute_Jaccard(dtboxes, gtboxes, bm_thr):

    assert dtboxes.shape[-1] > 3 and gtboxes.shape[-1] > 3
    if dtboxes.shape[0] < 1 or gtboxes.shape[0] < 1:
        return list()
    N, K = dtboxes.shape[0], gtboxes.shape[0]
    ious = compute_iou_matrix(dtboxes, gtboxes)
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches

def compute_JC(detection:np.ndarray, gt:np.ndarray, iou_thresh:np.ndarray):

    rows, cols = compute_lap(detection, gt, iou_thresh)
    return [(i, j) for i, j in zip(rows, cols)]

def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):

    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    eps = 1e-6
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)   
    ioas = inter / (dtarea + eps)
    return ioas

def compute_iou_matrix(dboxes:np.ndarray, gboxes:np.ndarray):
    
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    eps = 1e-6
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
    ious = inter / (dtarea + gtarea - inter + eps)
    return ious

def compute_maximal_iou(proposals:np.ndarray,gt:np.ndarray):
    
    ious = compute_iou_matrix(proposals, gt)
    return np.max(ious, axis = 1)

