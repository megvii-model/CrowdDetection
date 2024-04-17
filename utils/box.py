import os
import sys
import numpy as np
import math
from typing import List, Tuple
import pdb
class BoxView:
    """
    Utilities for box calculations.
    Create a view on the original data rather than make a copy

    Dimensions:
    (left, top, right, bottom)
    [left, right), [top, bottom)
    """

    _box = None

    def __init__(self, box):
        self._box = self._as_array(box)

    def __getitem__(self, index:int):
        return self._box[index]

    def __setitem__(self, index:int, v):
        self._box[index] = v

    def __len__(self):
        return 4

    def _as_array(self, box) -> np.ndarray:
        assert not (box is None)
        if isinstance(box, BoxView):
            return box.array
        else:
            assert isinstance(box, np.ndarray)
            assert box.size == 4
            return box.reshape([-1])

    def _as_readonly_array(self, box) -> np.ndarray:
        if isinstance(box, BoxView):
            return box.array
        elif isinstance(box, np.ndarray):
            assert box.size == 4
            return box.reshape([-1])
        elif box is None:
            return np.zeros([4])
        else:
            r = np.array(box)
            assert r.size == 4
            return r.reshape([-1])


    @property
    def array(self) -> np.ndarray:
        return self._box

    @array.setter
    def array(self, v):
        self._box = self._as_array(v)

    @property
    def width(self):
        return max(self._box[2] - self._box[0], 0)

    @property
    def height(self):
        return max(self._box[3] - self._box[1], 0)

    @property
    def valid(self):
        return self._box[2] > self._box[0] and self._box[3] > self._box[1]

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return np.float64(self._box[:2] + self._box[2:] - 1) / 2

    @property
    def vertices(self) -> np.ndarray:
        """
        return [left-top, left-bottom, right-bottom, right-top]
        """
        return np.array([
            (self._box[0], self._box[1]),
            (self._box[0], self._box[3] - 1),
            (self._box[2] - 1, self._box[3] - 1),
            (self._box[2] - 1, self._box[1])
            ])

    def intersect(self, other) -> "Box":
        other = self._as_readonly_array(other)
        left_top = np.max([self._box[:2], other[:2]], axis=0)
        right_bottom = np.min([self._box[2:], other[2:]], axis=0)
        r = Box(np.concatenate([left_top, right_bottom]))
        return r if r.valid else Box()

    def IoU(self, other) -> float:
        inter_area = self.intersect(other).area
        area_a = self.area
        area_b = Box(other).area
        if area_a == 0 and area_b == 0:
            return 0.
        else:
            return inter_area / (area_a + area_b - inter_area)

    def move_by(self, offset:np.ndarray):
        assert len(offset) == 2
        self._box[0] += offset[0]
        self._box[1] += offset[1]
        self._box[2] += offset[0]
        self._box[3] += offset[1]
        return self
    def move_by_points(self,offset:np.array):
        assert len(offset) == 4
        self._box[0] += offset[0]
        self._box[1] += offset[1]
        self._box[2] += offset[2]
        self._box[3] += offset[3]
        return self
    def is_legal(self):
        return self._box[0] > -1 and self._box[1] > -1 and self._box[2] > -1 \
            and self._box[3] > -1
    def clip_boundary(self,height,width):
        self._box[0] = np.max([0,self._box[0]])
        self._box[1] = np.max([0,self._box[1]])
        self._box[2] = np.min([width,self._box[2]])
        self._box[3] = np.min([height,self._box[3]])
        
    def expend_by(self, width:int):
        self._box[:2] -= width
        self._box[2:] += width
        return self

    def get_rotated_vertices(self, degree):
        """
        Get counter-clockwise rotated vertices of the box.
        The rotation center is the center of the box
        """
        vertices = np.float64(self.vertices)
        # pdb.set_trace()
        center = self.center
        center = center.reshape([1, -1])
        vertices -= center
        theta = degree / 180.0 * math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return np.dot(vertices, rot_mat.T) + center


class Box(BoxView):
    """
    Utilities for box calculations.
    Make a new copy from the original data

    Dimensions:
    (left, top, right, bottom)
    [left, right), [top, bottom)
    """

    @classmethod
    def create_tight_box(self, points:np.ndarray) -> "Box":
        points = np.int32(np.round(np.array(points)))
        if len(points.shape) <= 1:
            points = points.reshape([1, -1])

        assert len(points.shape) == 2
        assert points.shape[1] == 2

        left = points[:, 0].min()
        right = points[:, 0].max() + 1
        top = points[:, 1].min()
        bottom = points[:, 1].max() + 1
        return Box([left, top, right, bottom])

    @classmethod
    def create_tight_box_from_mask(self, mask:np.ndarray, value:np.array) -> "Box":
        mask = np.array(mask)
        value = np.array(value)

        assert len(mask.shape) == 2 or len(mask.shape) == 3
        mask = mask.reshape([mask.shape[0], mask.shape[1], -1])
        value = value.reshape([1, 1, -1])

        pos_h, pos_w = np.where(np.all(mask == value, axis=2))

        if len(pos_h) == 0:
            return Box()

        left = np.min(pos_w)
        top = np.min(pos_h)
        right = np.max(pos_w) + 1
        bottom = np.max(pos_h) + 1

        return Box([left, top, right, bottom])


    def __init__(self, box=None):
        super().__init__(box)

    def _as_array(self, box):
        """
        Override BoxView._as_array, create new copy
        """
        if isinstance(box, BoxView):
            return np.array(box.array)
        elif box is None:
            return np.zeros([4])
        elif box is np.ndarray:
            assert box.size == 4
            return np.array(box.flat)
        else:
            r = np.array(box)
            assert r.size == 4
            return r.reshape([-1])



# unit test

def _test1():
    a = Box([2,1,4,6])
    b = Box([3,5,4,9])
    c = Box([1,1,2,2])
    print(list(a.intersect(b)))
    print(a.IoU(b))
    print(a.IoU(a))
    print(a.IoU(c))
    print(list(Box()))

    data = np.array([3,2,4,5])
    d = BoxView(data)
    pdb.set_trace()
    print(data)
    d[2] = 44
    print(list(d))
    print(data)

    print(len(d))

def _test2():
    from .draw import fill_box, Color
    import cv2 as cv

    img = np.zeros([600, 400, 3])
    box = Box([230, 100, 400, 200])

    fill_box(box, img, Color.Red)

    b = Box.create_tight_box_from_mask(img, Color.Red)
    print(list(b))


if __name__ == '__main__':
    _test1()
