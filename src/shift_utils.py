import torch
import numpy as np
from torch import nn
from abc import ABCMeta, abstractmethod

class ShiftBase(torch.nn.Module,metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x_fourier, shift_params):
        ...

class ShiftIdentity(ShiftBase):
    def __init__(self):
        super(ShiftIdentity,self).__init__()

    def forward(self, x_fourier, idcs=0, shift_params={}, mode='none', frequency_marcher=None):
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]  # adds a batch dimension so as to be compatible with our other CTFs
        return x_fourier

class Shift(ShiftBase):
    """
    A class containing method to shift an image in Fourier domain.
    ...
    Attributes
    ----------
    size : int
            side length of images in pixels
    resolution : float
            physical size of the pixels in Angstrom (A)
    Methods
    -------
    forward(x_fourier, shift_params={}):
    outputs shifted x_fourier depending on shift_params.

    modulate(x_fourier, t_x,t_y):
    outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
    The modulation depends on t_x, t_y.

    """
    def __init__(self, num_particles, size=128, resolution=0.8, requires_grad=False, std_shift=5.0, flip_images=False):
        """
        Initialization of a Shift object.

        Parameters
        ----------
        num_particles: int
        size: int
        resolution: float
        requires_grad: bool
        std_shift: float
        flip_images: bool
        """
        super(Shift, self).__init__()
        self.requires_grad = requires_grad
        self.flip_images = flip_images

        self.size = size
        self.resolution = resolution
        self.frequency = 1. / (self.size * resolution)

        n2 = float(self.size // 2)
        ax = torch.arange(-n2, n2 + self.size % 2)
        ax = torch.flip(ax, dims=[0])
        mx, my = torch.meshgrid(ax, ax)

        # shape SizexSize
        self.register_buffer("mx", mx)
        self.register_buffer("my", my)

        # Generate shifts
        self.generate_shifts(std_shift, num_particles)

    def modulate(self, x_fourier, t_x, t_y):
        """
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.

        Parameters
        ----------
        x_fourier : torch.Tensor (Bx1xSizexSize)
            batch of input images in Fourier domain
        t_x: torch.Tensor (B,)
            batch of shifts along horizontal axis in A
        t_y: torch.Tensor (B,)
            batch of shifts along vertical axis in A
        Returns
        -------
        output: torch.Tensor (Bx1xSizexSize)
            batch of modulated fourier images given by
            output(f_1,f_2)=e^{-2*pi*j*[f_1,f_2]*[t_x, t_y] }*input(f_1,f_2)
        """
        t_y = t_y[:, None, None, None] # [B,1,1,1]
        t_x = t_x[:, None, None, None] # [B,1,1,1]

        modulation = torch.exp(-2 * np.pi * 1j * self.frequency * (self.mx_after_fm * t_y +
                                                                   self.my_after_fm * t_x)) # [B,1,Size,Size]

        return x_fourier * modulation # [B,1,Size,Size]*[B,1,Size,Size]

    def update_mx_my(self, frequency_marcher):
        if frequency_marcher is not None:
            size = 2 * frequency_marcher.f + 1
            if size > self.size:
                size = self.size
            self.mx_after_fm = frequency_marcher.cut_coords_plane(
                self.mx.reshape(self.size, self.size, 1)
            ).reshape(size, size)
            self.my_after_fm = frequency_marcher.cut_coords_plane(
                self.my.reshape(self.size, self.size, 1)
            ).reshape(size, size)
        else:
            self.mx_after_fm = self.mx
            self.my_after_fm = self.my

    def forward(self, x_fourier, idcs=0, shift_params={}, mode='gt', frequency_marcher=None):
        """
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.

        Parameters
        ----------
        x_fourier: torch.Tensor (Bx1xSizexSize)
            batch of input images in Fourier domain
        idcs: indices (list)

        shift_params:
            dictionary containing
            'shiftX': torch.Tensor (B,)
                batch of shifts along horizontal axis
            'shiftY': torch.Tensor (B,)
                batch of shifts along vertical axis
        Returns
        -------
        output: torch.Tensor (Bx1xSizexSize)
            batch of modulated fourier images if shift_params is not empty else input is outputted
        """
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]

        self.update_mx_my(frequency_marcher)

        batch_size = len(idcs)
        if shift_params:  # training mode
            x_fourier_shifted = self.modulate(x_fourier[:batch_size],
                                              shift_params['shiftX'][:batch_size],
                                              shift_params['shiftY'][:batch_size])  # B, 1, S, S
        else:  # simulation mode
            x_fourier_shifted = self.modulate(x_fourier[:batch_size],
                                              self.shifts[idcs, 0],
                                              self.shifts[idcs, 1])

        if self.flip_images:
            if mode == 'gt':
                x_fourier_anti_shifted = self.modulate(x_fourier[batch_size:],
                                                       -shift_params['shiftX'],
                                                       -shift_params['shiftY'])
            else:
                assert shift_params, " shift params should not be empty here"
                x_fourier_anti_shifted = self.modulate(x_fourier[batch_size:],
                                                       shift_params['shiftX'][batch_size:],
                                                       shift_params['shiftY'][batch_size:])
            x_fourier_shifted = torch.cat([x_fourier_shifted, x_fourier_anti_shifted], dim=0)

        return x_fourier_shifted

    def generate_shifts(self, std_dev, num_particles):
        """
        std_dev: standard deviation of the shift in A
        num_particles: number of particles
        """
        shifts = std_dev * torch.randn((num_particles, 2))
        # Limit the shift at 5 std
        norms = torch.norm(shifts, dim=1)
        too_far = (norms > 5 * std_dev + 1e-3)
        shifts[too_far] = 5 * std_dev * shifts[too_far] / norms[too_far].reshape(-1, 1)

        self.shifts = nn.Parameter(shifts, requires_grad=self.requires_grad)  # N, 2