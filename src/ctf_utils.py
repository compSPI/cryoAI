import scipy.constants
import torch
from torch import nn
import math
import numpy as np
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F


def primal_to_fourier_2D(r):
    r = torch.fft.ifftshift(r, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))


def primal_to_fourier_3D(r):
    r = torch.fft.ifftshift(r, dim=(-3, -2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                              dim=(-3, -2, -1))


def fourier_to_primal_2D(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))


''' CTF abstract class'''
class CTFBase(nn.Module,metaclass=ABCMeta):
    def __init__(self, resolution, num_particles,
                 requires_grad=False):
        super(CTFBase, self).__init__()
        self.resolution = resolution
        self.requires_grad = requires_grad

    @abstractmethod
    def forward(self, x_fourier, idcs=0, ctf_params={}, mode='gt', frequency_marcher=None):
        ...


class CTFRelion(CTFBase):
    def __init__(self, size=257, resolution=0.8,
                 kV=300.0, valueNyquist=.001,
                 defocusU=1., defocusV=1., sim_ctf_defocus_stdev=0.1, angleAstigmatism=0.,
                 cs=2.7, phasePlate=0., amplitudeContrast=.1,
                 bFactor=0., num_particles=500,
                 requires_grad=False, precompute=True, flip_images=False, device='cuda', variable_ctf=False):
        """
        Initialization of CTF object with Relion conventions.

        Parameters
        ----------
        size: int
        resolution: float
        kV: float
        valueNyquist: float
        defocusU: float
        defocusV: float
        sim_ctf_defocus_stdev: float
        angleAstigmatism: float
        cs: float
        phasePlate: float
        amplitudeContrast: float
        bFactor: float
        num_particles: int
        requires_grad: bool
        precompute: bool
        flip_images: bool
        device: str
        variable_ctf: bool
        """
        super(CTFRelion, self).__init__(resolution, num_particles, requires_grad)
        self.requires_grad = requires_grad
        self.flip_images = flip_images
        self.device = device
        self.variable_ctf = variable_ctf

        if variable_ctf:
            assert self.device == 'cuda', "Variable CTF only implemented on GPU."
        if requires_grad:
            assert self.device == 'cuda', "The CTF cannot be differentiable if not on GPU."

        self.size = size  # in pixel
        self.resolution = resolution  # in angstrom
        self.kV = kV  # in kilovolt

        self.valueNyquist = valueNyquist
        self.phasePlate = phasePlate/180. * np.pi  # in radians (converted from degrees)
        self.amplitudeContrast = amplitudeContrast
        self.bFactor = bFactor

        self.frequency = 1./self.resolution

        self.wavelength = self._get_ewavelength(self.kV * 1e3)  # input in V (so we convert kv*1e3)

        angleAstigmatism = angleAstigmatism / 180. * np.pi  # input in degree converted in radian
        cs = cs * 1e7  # input in mm converted in angstrom
        # If device is 'cuda', the angleAstigmatism, defocusU, defocusV and cs are nn.Parameter of size (N, 1, 1)
        # (on GPU), otherwise they are tensors of size (1), overwritten by tensors of size (B, 1, 1) in encoder mode
        if device == 'cuda':
            self.angleAstigmatism = nn.Parameter(
                angleAstigmatism * torch.ones((num_particles, 1, 1), dtype=torch.float32),
                requires_grad=requires_grad)
            self.cs = nn.Parameter(cs * torch.ones((num_particles, 1, 1), dtype=torch.float32),
                                   requires_grad=requires_grad)
            if variable_ctf:
                assert np.abs(defocusU - defocusV) < 1e-3, "defocusU and defocusV must be identical with variable CTF."
                defocii = np.random.lognormal(np.log(defocusU),
                                              sim_ctf_defocus_stdev,
                                              num_particles).reshape(num_particles, 1, 1)
                self.defocusU = nn.Parameter(torch.tensor(defocii, dtype=torch.float32),
                                             requires_grad=requires_grad)
                self.defocusV = nn.Parameter(torch.tensor(defocii, dtype=torch.float32),
                                             requires_grad=requires_grad)
            else:
                self.defocusU = nn.Parameter(defocusU * torch.ones((num_particles,1,1), dtype=torch.float32),
                                             requires_grad=requires_grad)
                self.defocusV = nn.Parameter(defocusV * torch.ones((num_particles,1,1), dtype=torch.float32),
                                             requires_grad=requires_grad)
        else:
            self.angleAstigmatism = torch.tensor([angleAstigmatism])
            self.defocusU = torch.tensor([defocusU])
            self.defocusV = torch.tensor([defocusV])
            self.cs = torch.tensor([cs])

        self.precomputed_filters = precompute

        ax = torch.linspace(-1./(2.*resolution), 1/(2.*resolution), self.size)
        mx, my = torch.meshgrid(ax, ax)
        self.register_buffer("r2", mx ** 2 + my ** 2)
        self.register_buffer("r", torch.sqrt(self.r2))
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

        if not self.requires_grad and self.precomputed_filters and self.device == 'cuda':
            print("Precomputing hFourier in CTF")
            self.register_buffer('hFourier', self.get_ctf(torch.arange(num_particles), num_particles))

    def _get_ewavelength(self,U):
        # assumes V as input, returns wavelength in angstrom
        h = scipy.constants.h
        e = scipy.constants.e
        c = scipy.constants.c
        m0 = scipy.constants.m_e

        return h / math.sqrt(2.*m0*e*U)/math.sqrt(1+e*U/(2*m0*c**2)) * 1e10

    def get_psf(self, idcs):
        hFourier = self.get_ctf(idcs)
        hSpatial = torch.fft.fftshift(
                        torch.fft.ifftn(
                            torch.fft.ifftshift(hFourier,
                                                dim=(-2,-1)),
                                        s=(hFourier.shape[-2],hFourier.shape[-1]),
                                        dim=(-2,-1))) # is complex
        return hSpatial

    def get_ctf(self, idcs, B, cpu_params={}, frequency_marcher=None):
        if self.device == 'cuda':
            defocusU = self.defocusU[idcs, :, :]
            defocusV = self.defocusV[idcs, :, :]
            angleAstigmatism = self.angleAstigmatism[idcs, :, :]
            cs = self.cs[idcs, :, :]
        else:
            if cpu_params:
                defocusU = cpu_params['defocusU'].to(self.angleFrequency.device)
                defocusV = cpu_params['defocusV'].to(self.angleFrequency.device)
                angleAstigmatism = cpu_params['angleAstigmatism'].to(self.angleFrequency.device)
            else:
                defocusU = self.defocusU.reshape(1, 1, 1).repeat(B, 1, 1).to(self.angleFrequency.device)
                defocusV = self.defocusV.reshape(1, 1, 1).repeat(B, 1, 1).to(self.angleFrequency.device)
                angleAstigmatism = self.angleAstigmatism.reshape(1, 1, 1).repeat(B, 1, 1).to(self.angleFrequency.device)
            cs = self.cs.reshape(1, 1, 1).repeat(B, 1, 1).to(self.angleFrequency.device)

        ac = self.amplitudeContrast
        pc = math.sqrt(1.-ac**2)
        K1 = np.pi/2. * cs * self.wavelength**3
        K2 = np.pi * self.wavelength

        # Cut-off from frequency marcher
        if frequency_marcher is not None:
            self.size_after_fm = 2 * frequency_marcher.f + 1
            if self.size_after_fm > self.size:
                self.size_after_fm = self.size
            angleFrequency = frequency_marcher.cut_coords_plane(
                self.angleFrequency.reshape(self.size, self.size, 1)
            ).reshape(self.size_after_fm, self.size_after_fm)
            r2 = frequency_marcher.cut_coords_plane(
                self.r2.reshape(self.size, self.size, 1)
            ).reshape(self.size_after_fm, self.size_after_fm)
        else:
            self.size_after_fm = self.size
            angleFrequency = self.angleFrequency
            r2 = self.r2

        angle = angleFrequency - angleAstigmatism
        local_defocus =   1e4*(defocusU + defocusV)/2. \
                        + angleAstigmatism * torch.cos(2.*angle)

        gamma = K1 * r2**2 - K2 * r2 * local_defocus - self.phasePlate
        hFourier = -pc*torch.sin(gamma) + ac*torch.cos(gamma)

        if self.valueNyquist != 1:
            decay = np.sqrt(-np.log(self.valueNyquist)) * 2. * self.resolution
            envelope = torch.exp(-self.frequency * decay ** 2 * r2)
            hFourier *= envelope

        return hFourier

    def oversample_multiply_crop(self, x_fourier, hFourier):
        # we assume that the shape of the CTF is always going to be bigger
        # than the size of the input image
        input_sz = x_fourier.shape[-1]
        if input_sz != self.size_after_fm:
            x_primal = fourier_to_primal_2D(x_fourier)

            pad_len = (self.size_after_fm - x_fourier.shape[-1])//2 # here we assume even lengths
            p2d = (pad_len,pad_len,pad_len,pad_len)
            x_primal_padded = F.pad(x_primal,p2d,'constant',0)

            x_fourier_padded = primal_to_fourier_2D(x_primal_padded)

            x_fourier_padded_filtered = x_fourier_padded * hFourier[:, None, :, :]
            return x_fourier_padded_filtered[..., pad_len:-pad_len, pad_len:-pad_len]
        else:
            return x_fourier * hFourier[:, None, :, :]


    def get_cpu_params(self, idcs, ctf_params, flip=False):
        batch_size = idcs.shape[0]
        if self.device == 'cuda':
            self.defocusU[idcs, :, :] = ctf_params['defocusU'][:batch_size] if not flip else\
                ctf_params['defocusU'][batch_size:]
            self.defocusV[idcs, :, :] = ctf_params['defocusV'][:batch_size] if not flip else\
                ctf_params['defocusV'][batch_size:]
            self.angleAstigmatism[idcs, :, :] = ctf_params['angleAstigmatism'][:batch_size] if not flip else\
                ctf_params['angleAstigmatism'][batch_size:]
            cpu_params = {}
        else:
            defocusU = ctf_params['defocusU'][:batch_size] if not flip else ctf_params['defocusU'][batch_size:]
            defocusV = ctf_params['defocusV'][:batch_size] if not flip else ctf_params['defocusV'][batch_size:]
            angleAstigmatism = ctf_params['angleAstigmatism'][:batch_size] if not flip else\
                ctf_params['angleAstigmatism'][batch_size:]
            cpu_params = {'defocusU': defocusU.reshape(batch_size, 1, 1),
                          'defocusV': defocusV.reshape(batch_size, 1, 1),
                          'angleAstigmatism': angleAstigmatism.reshape(batch_size, 1, 1)}
        return cpu_params


    def forward(self, x_fourier, idcs=0, ctf_params={}, mode='gt', frequency_marcher=None):
        # This is when we want to prescribe parameters for the CTF
        if x_fourier.dim() == 3:
            x_fourier = x_fourier[None, ...]
        # x_fourier: B, 1, S, S
        batch_size = len(idcs)
        cpu_params = {}
        if ctf_params:
            cpu_params = self.get_cpu_params(idcs, ctf_params, flip=False)

        # if new params for the CTF have been prescribed or we are optimizing it
        # then request the evaluation of the CTF
        if self.device == 'cuda' and not ctf_params and self.precomputed_filters and not self.requires_grad:
            hFourier = self.hFourier[idcs, :, :]
        else:
            hFourier = self.get_ctf(idcs, batch_size, cpu_params=cpu_params, frequency_marcher=frequency_marcher)


        if self.flip_images:
            flipped_hFourier = torch.flip(hFourier, [1, 2])

            hFourier = torch.cat([hFourier, flipped_hFourier], dim=0)

        return self.oversample_multiply_crop(x_fourier, hFourier)
