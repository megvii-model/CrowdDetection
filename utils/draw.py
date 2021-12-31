#coding:utf-8
import numpy as np
import cv2
from typing import Tuple
from box import Box
import pdb
class Color:
    Red = np.array([0, 0, 255])
    Green = np.array([0, 255, 0])
    Blue = np.array([255, 0, 0])
    White = np.array([255, 255, 255])
    Black = np.array([0, 0, 0])
    Gray = np.array([128, 128, 128])
    Yellow = np.array([255, 215, 0])
    Brown = np.array([128, 42, 42])

def draw_box(box:Box, target:np.ndarray, color, line_width:int=1):
    assert line_width > 0
    width_outer = line_width // 2
    assert len(target.shape) == 2 or len(target.shape) == 3
    target = target.reshape([target.shape[0], target.shape[1], -1])

    width = target.shape[1]
    height = target.shape[0]

    box_outer = Box(box).expend_by(width_outer)

    if box_outer.valid:
        # top line
        h_start = max(0, box_outer[1])
        h_end = min(height, box_outer[1] + line_width, box_outer[3])
        w_start = max(0, box_outer[0])
        w_end = min(width, box_outer[2])
        if h_start < h_end and w_start < w_end:
            target[h_start:h_end, w_start:w_end, :] = color

        # bottom line
        h_start = max(box_outer[1], box_outer[3] - line_width, 0)
        h_end = min(height, box_outer[3])
        if h_start < h_end:
            target[h_start:h_end, w_start:w_end, :] = color

        # left line
        h_start = max(0, box_outer[1])
        h_end = min(height, box_outer[3])
        w_start = max(0, box_outer[0])
        w_end = min(width, box_outer[0] + line_width, box_outer[2])
        if h_start < h_end and w_start < w_end:
            target[h_start:h_end, w_start:w_end, :] = color

        # right line
        w_start = max(box_outer[0], box_outer[2] - line_width, 0)
        w_end = min(width, box_outer[2])
        if w_start < w_end:
            target[h_start:h_end, w_start:w_end, :] = color



def fill_box(box:Box, target:np.ndarray, color):
    assert len(target.shape) == 2 or len(target.shape) == 3
    target = target.reshape([target.shape[0], target.shape[1], -1])

    width = target.shape[1]
    height = target.shape[0]

    box = Box(box).intersect(Box([0, 0, width, height]))
    if box.valid:
        target[box[1]:box[3], box[0]:box[2], :] = color
        
def draw_xt(xt:np.ndarray,image:np.ndarray,color:Color,line_width:int):
        assert image is not None
        xt = np.int32(np.round(xt[:,:4]))
        nr_xt = xt.shape[0]
        for i in range(nr_xt):
                box = Box(xt[i,:])
                draw_box(box,image,color,line_width)
def fill_poly(points:np.ndarray, target:np.ndarray, color):
    points = np.int32(np.round(np.array(points)))
    assert len(points.shape) == 2
    assert points.shape[1] == 2

    assert len(target.shape) == 2 or len(target.shape) == 3
    target = target.reshape([target.shape[0], target.shape[1], -1])

    height = target.shape[0]
    width = target.shape[1]

    n = points.shape[0]
    p1 = np.hstack([points, np.ones([n, 1], dtype=np.int32)])
    p2 = np.vstack([points[1:, :], points[:1, :]])
    p2 = np.hstack([p2, np.ones([n, 1], dtype=np.int32)])
    edges = np.cross(p1, p2)

    tight_box = Box.create_tight_box(points).intersect(Box([0, 0, width, height]))
    iy, ix = np.mgrid[tight_box[1]:tight_box[3], tight_box[0]:tight_box[2]]
    iy = iy.reshape([-1, 1])
    ix = ix.reshape([-1, 1])
    inner_points = np.hstack([ix, iy, np.ones([iy.size, 1], dtype=np.int32)])
    r = np.dot(inner_points, edges.T)
    r = (r <= 0).all(axis=1)
    if r.size > 0:
        inner_points = inner_points[r, :]
        target[inner_points[:, 1], inner_points[:, 0], :] = color


# unit test
def _unit_test():
    import cv2 as cv

    b = Box([-10, 400, 300, 600])
    img = np.zeros([1000, 1000, 3])

    draw_box(b, img, Color.Green, 2)
    fill_box(b, img, Color.Red)

    #fill_poly(b.vertices, img, Color.Blue)
    #ss = np.int64(b.get_rotated_vertices(45))
    #print(ss)
    #fill_poly(ss, img, Color.Blue)

    cv.imshow("test", img)
    cv.waitKey(0)

if __name__ == '__main__':
    _unit_test()

