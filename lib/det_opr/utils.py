import megengine as mge
import megengine.functional as F
import numpy as np

from megengine.core import Tensor

def get_padded_tensor(
    array: Tensor, multiple_number: int = 32, pad_value: float = 0
) -> Tensor:
    """ pad the nd-array to multiple stride of th e

    Args:
        array (Tensor):
            the tensor with the shape of [batch, channel, height, width]
        multiple_number (int):
            make the height and width can be divided by multiple_number
        pad_value (int): the value to be padded

    Returns:
        padded_array (Tensor)
    """
    batch, chl, t_height, t_width = array.shape
    padded_height = (
        (t_height + multiple_number - 1) // multiple_number * multiple_number
    )
    padded_width = (t_width + multiple_number - 1) // multiple_number * multiple_number

    padded_array = (
        mge.ones(
            F.concat([batch, chl, padded_height, padded_width], axis=0),
            dtype=np.float32,
        )
        * pad_value
    )

    ndim = array.ndim
    if ndim == 4:
        padded_array = padded_array.set_subtensor(array)[:, :, :t_height, :t_width]
    elif ndim == 3:
        padded_array = padded_array.set_subtensor(array)[:, :t_height, :t_width]
    else:
        raise Exception("Not supported tensor dim: %d" % ndim)
    return padded_array

from megengine.core.tensor import wrap_io_tensor
import megengine._internal as mgb
@wrap_io_tensor
def cond_take(data, mask, **kwargs):
    return mgb.opr.cond_take(data, mask, **kwargs)

def mask_to_inds(mask):
    _, inds = cond_take(mask, mask, mode=mgb.opr_param_defs.CondTake.Mode.EQ, val=1)
    return F.zero_grad(inds)
