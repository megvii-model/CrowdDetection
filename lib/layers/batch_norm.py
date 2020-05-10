import numpy as np
import megengine.module as M
from megengine.core import Buffer

class FrozenBatchNorm2d(M.Module):
    """
    BatchNorm2d, which the weight, bias, running_mean, running_var
    are immutable.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Buffer(np.ones(num_features, dtype=np.float32))
        self.bias = Buffer(np.zeros(num_features, dtype=np.float32))
        self.running_mean = Buffer(np.zeros((1, num_features, 1, 1), dtype=np.float32))
        self.running_var = Buffer(np.ones((1, num_features, 1, 1), dtype=np.float32))
    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1) * (1.0 / (self.running_var + self.eps).sqrt())
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean * scale
        return x * scale + bias

