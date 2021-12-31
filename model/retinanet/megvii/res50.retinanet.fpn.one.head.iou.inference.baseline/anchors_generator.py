import numpy as np
import torch
import torch.nn.functional as F
import pdb
class AnchorGenerator():
    """default anchor generator for fpn.
    This class generate anchors by feature map in level.
    """
    def __init__(self, base_size=16, ratios=[0.5, 1, 2],
        base_scale=2):
        self.base_size = base_size
        self.base_scale = np.array(base_scale)
        self.anchor_ratios = np.array(ratios)

    def _whctrs(self, anchor):
        """convert anchor box into (w, h, ctr_x, ctr_y)
        """
        w = anchor[:, 2] - anchor[:, 0] + 1
        h = anchor[:, 3] - anchor[:, 1] + 1
        x_ctr = anchor[:, 0] + 0.5 * (w - 1)
        y_ctr = anchor[:, 1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr
    
    def _ratio_enum(self, anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors
    
    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """

        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    def _get_cell_anchors(self, anchor_scales):

        base_anchor = np.array([1, 1, self.base_size, self.base_size]) - 1
        ratio_anchors = self._ratio_enum(base_anchor, self.anchor_ratios)
        anchors = np.vstack([self._scale_enum(ratio_anchors[i, :], anchor_scales)
                            for i in range(ratio_anchors.shape[0])])
        return anchors.astype(np.float32)

    def _scale_enum(self, anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def get_plane_anchors(self, anchor_scales: np.ndarray):
        """get anchors per location on feature map.
        The anchor number is anchor_scales x anchor_ratios
        """
        base_anchor = np.array([[0, 0, self.base_size - 1, self.base_size - 1]])
        off = self.base_size // 2 - 8
        w, h, x_ctr, y_ctr = self._whctrs(base_anchor)
        # ratio enumerate
        size = w * h
        size_ratios = size / self.anchor_ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * self.anchor_ratios)
        # scale enumerate
        anchor_scales = anchor_scales[None, ...]
        ws = (ws[:, None] * anchor_scales).reshape(-1, 1)
        hs = (hs[:, None] * anchor_scales).reshape(-1, 1)
        # make anchors
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1))) - off
        all_anchors = self._get_cell_anchors(anchor_scales)
        anchors = np.vstack([anchors, all_anchors])
        return anchors.astype(np.float32)

    def get_center_offsets(self, fm_map, stride):
        # fm_height, fm_width = fm_map.shape[-2], fm_map.shape[-1]
        fm_height, fm_width = fm_map.shape[2:]
        f_device = fm_map.device
        shift_x = torch.arange(0, fm_width, device=f_device) * stride
        shift_y = torch.arange(0, fm_height, device=f_device) * stride
        broad_shift_x = shift_x.reshape(-1, shift_x.shape[0]).repeat(fm_height,1)
        broad_shift_y = shift_y.reshape(shift_y.shape[0], -1).repeat(1,fm_width)
        flatten_shift_x = broad_shift_x.flatten().reshape(-1,1)
        flatten_shift_y = broad_shift_y.flatten().reshape(-1,1)
        shifts = torch.cat(
            [flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y],
            axis=1)
        return shifts 

    def get_anchors_by_feature(self, fm_map, base_stride, off_stride):
        # shifts shape: [A, 4]
        shifts = self.get_center_offsets(fm_map, base_stride * off_stride)
        # plane_anchors shape: [B, 4], e.g. B=3
        plane_anchors = self.get_plane_anchors(self.base_scale * off_stride)
        plane_anchors = torch.tensor(plane_anchors, device=fm_map.device)
        # all_anchors shape: [A, B, 4]
        all_anchors = plane_anchors[None, :] + shifts[:, None]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    @torch.no_grad()
    def __call__(self, featmap, base_stride, off_stride):
        return self.get_anchors_by_feature(featmap, base_stride, off_stride)

if __name__ == '__main__':

    instance = AnchorGenerator(base_size=16, ratios = [1, 2, 3])
    anchors = instance.get_plane_anchors(anchor_scales = np.array([16]))
    # [[-120. -120.  135.  135.]
    #  [ -80. -168.   95.  183.]
    #  [ -64. -208.   79.  223.]]
    print(anchors)
