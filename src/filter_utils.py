import torch
import torch.nn as nn

class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size, kernel_variance, num_octaves, octave_scaling):
        """
        Initialize a set of gaussian filters.

        Parameters
        ---------
        kernel_size: int
        kernel_variance: float
        num_octaves: int
        octave_scaling: int
        """
        super(GaussianPyramid,self).__init__()
        self.kernel_size = kernel_size
        self.variance = kernel_variance
        self.num_dec = num_octaves
        self.scaling = octave_scaling

        weighting = torch.ones([num_octaves], dtype=torch.float32)
        self.register_buffer('weighting', weighting)
        self.kernels = self.generateGaussianKernels(kernel_size, kernel_variance, num_octaves + 1, octave_scaling)

        self.gaussianPyramid = torch.nn.Conv2d(1, num_octaves + 1,
                                               kernel_size=kernel_size,
                                               padding='same', padding_mode='reflect', bias=False)
        self.gaussianPyramid.weight = torch.nn.Parameter(self.kernels)
        self.gaussianPyramid.weight.requires_grad = False

    def generateGaussianKernels(self, size, var, scales=1, scaling=2):
        """
        Generate a list of gaussian kernels

        Parameters
        ----------
        size: int
        var: float
        scales: int
        scaling: int

        Returns
        -------
        kernels: list of torch.Tensor
        """
        coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xy = torch.stack(torch.meshgrid(coords, coords),dim=0)
        kernels = [torch.exp(-(xy ** 2).sum(0) / (2 * var * scaling ** i)) for i in range(scales)]
        kernels = torch.stack(kernels,dim=0)
        kernels /= kernels.sum((1, 2), keepdims=True)

        kernels = kernels[:, None, ...]
        return kernels

    def forward(self, x):
        return self.gaussianPyramid(x)
