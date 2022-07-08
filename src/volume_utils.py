import torch
from torch import nn
import torch.fft
from abc import ABCMeta, abstractmethod

from .ml_modules import SIREN, FCBlock, PositionalEncoding, FourierNet


class VolumeBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def make_volume(self):
        ...


def shift_coords(coords, x_range, y_range, z_range, Nx, Ny, Nz, flip=False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).

    Parameters
    ----------
    coords: torch.tensor (..., 3)
    x_range: float
        (max_x - min_x) / 2
    y_range: float
        (max_y - min_y) / 2
    z_range: float
        (max_z - min_z) / 2
    Nx: int
    Ny: int
    Nz: int
    flip: bool

    Returns
    -------
    coords: torch.tensor (..., 3)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.

    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz % 2 == 0:
        z_shift = coords[..., 2] + alpha * z_range / (Nz - 1)
    else:
        z_shift = coords[..., 2]
    coords = torch.cat((x_shift.unsqueeze(-1),
                        y_shift.unsqueeze(-1),
                        z_shift.unsqueeze(-1)), dim=-1)
    return coords


class ImplicitFourierVolume(VolumeBase):
    def __init__(self, img_sz, params_implicit, frequency_marcher):
        """
        Initialization of an implicit representation of the volume in Fourier space.

        Parameters
        ----------
        img_sz: int
        params_implicit: dictionary
        frequency_marcher: FrequencyMarcher
        """
        super(ImplicitFourierVolume,self).__init__()
        self.img_sz = img_sz
        self.frequency_marcher = frequency_marcher
        self.is_chiral = False  # boolean that tells us if the current representation is the chiral transform of gt

        lincoords = torch.linspace(-1., 1., self.img_sz)
        [X, Y] = torch.meshgrid([lincoords, lincoords])
        coords = torch.stack([Y, X, torch.zeros_like(X)], dim=-1)
        coords = shift_coords(coords, 1., 1., 0, img_sz, img_sz, 1)
        self.register_buffer('plane_coords', coords.reshape(-1, 3))

        lincoords = torch.linspace(-1., 1., self.img_sz)
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        coords = torch.stack([Y, X, Z], dim=-1)
        coords = shift_coords(coords, 1., 1., 1., img_sz, img_sz, img_sz)
        self.register_buffer('coords_3d', coords.reshape(-1, 3))

        if params_implicit["type"] == 'siren':
            self.fvol = SIREN(in_features=3, out_features=2,
                              num_hidden_layers=4, hidden_features=256,
                              outermost_linear=True, w0=30)
            self.pe = None
        elif params_implicit["type"] == 'fouriernet':
            self.fvol = FourierNet(force_symmetry=params_implicit['force_symmetry'])
            self.pe = None
        elif params_implicit["type"] == 'relu_pe':
            num_encoding_fns = 6
            self.pe = PositionalEncoding(num_encoding_fns)
            self.fvol = FCBlock(in_features=3*(1+2*num_encoding_fns), features=[256, 256, 256, 256],
                                out_features=2, nonlinearity='relu', last_nonlinearity=None)
        else:
            raise NotImplementedError

    def forward(self, rotmat):
        """
        Generates a slice in Fourier space from a rotation matrix.

        Parameters
        ----------
        rotmat: torch.Tensor (B, 3, 3)

        Returns
        -------
        fplane: torch.Tensor (B, 1, img_sz, img_sz) (complex)
        """
        batch_sz = rotmat.shape[0]

        if self.frequency_marcher is not None:
            plane_coords = self.frequency_marcher.cut_coords_plane(self.plane_coords)
            img_sz = self.frequency_marcher.n_freq
        else:
            plane_coords = self.plane_coords
            img_sz = self.img_sz

        rot_plane_coords = torch.bmm(plane_coords.repeat(batch_sz, 1, 1), rotmat)  # B, img_sz^2, 3

        if self.pe is not None:
            rot_plane_coords = self.pe(rot_plane_coords)

        fplane = self.fvol(rot_plane_coords).reshape(batch_sz, img_sz, img_sz, 2)

        fplane = torch.view_as_complex(fplane)

        fplane = fplane[:, None, :, :]
        return fplane

    def make_volume(self, resolution='full'):
        """
        Generates a voxel-grid volume.

        Parameters
        ----------
        resolution: str / int

        Returns:
        output: torch.Tensor (img_sz, img_sz, img_sz)
        """
        with torch.no_grad():
            if resolution == 'full':
                coords = self.coords_3d
                img_sz = self.img_sz
            else:
                left = max(0, self.img_sz // 2 - resolution)
                right = min(self.img_sz, self.img_sz // 2 + resolution + 1)
                coords = self.coords_3d.reshape(self.img_sz,
                                                self.img_sz,
                                                self.img_sz, 3)[left:right, left:right, left:right, :].reshape(-1, 3)
                img_sz = right - left

            if self.pe is not None:
                coords = self.pe(coords)

            exp_fvol = self.fvol(coords).reshape(img_sz, img_sz, img_sz, 2)

            exp_fvol = torch.view_as_complex(exp_fvol)
            exp_fvol = torch.fft.ifftshift(exp_fvol, dim=(-3, -2, -1))

            exp_vol = torch.fft.fftshift(torch.fft.ifftn(exp_fvol, s=(img_sz, img_sz, img_sz),
                                                         dim=(-3, -2, -1)),
                                         dim=(-3, -2, -1))
            return exp_vol.real