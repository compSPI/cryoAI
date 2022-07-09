import torch
import torch.fft
import mrcfile
import starfile
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
import numpy as np
import os
from .geom_utils import euler_angles2matrix
from .ctf_utils import primal_to_fourier_2D, primal_to_fourier_3D, fourier_to_primal_2D
from .mask_utils import Mask
from .volume_utils import shift_coords
from pytorch3d.transforms import random_rotations


def get_power(vol):
    return(np.sum(np.abs(vol)))


class StarfileDataLoader(Dataset):
    def __init__(self, config):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        config: namespace
        """
        super(StarfileDataLoader, self).__init__()

        resize_input = config.side_len
        mask_rad = config.mask_rad
        sidelen_input = config.side_len * config.side_len_input_output_ratio

        self.path_to_starfile = config.path_to_starfile
        self.starfile = config.starfile
        self.df = starfile.open(os.path.join(self.path_to_starfile, self.starfile))
        self.sidelen_input = sidelen_input
        self.scale_images = config.scale_images
        self.no_trans = config.no_trans

        self.resize_input = resize_input
        if self.resize_input is None:
            self.vol_sidelen = self.df['optics']['rlnImageSize'][0]
            self.true_sidelen = self.vol_sidelen
        else:
            self.vol_sidelen = resize_input
            self.true_sidelen = self.df['optics']['rlnImageSize'][0]
        self.invert_hand = config.invert_hand

        config.map_shape = [self.vol_sidelen] * 3

        assert config.idx_max < len(self.df['particles']), "idx_max is too high."
        assert config.idx_min >= 0, "idx_min is too low."
        if config.idx_max < 0:
            config.idx_max = len(self.df['particles']) - 1
        config.num_particles = config.idx_max - config.idx_min + 1
        self.num_projs = config.idx_max - config.idx_min + 1
        self.idx_min = config.idx_min

        config.ctf_size = self.vol_sidelen
        config.kV = self.df['optics']['rlnVoltage'][0]
        config.resolution = self.df['optics']['rlnImagePixelSize'][0] * self.true_sidelen / self.vol_sidelen
        config.spherical_aberration = self.df['optics']['rlnSphericalAberration'][0]
        config.amplitude_contrast = self.df['optics']['rlnAmplitudeContrast'][0]

        self.mask_rad = mask_rad
        if self.mask_rad > 1e-3:
            self.mask = Mask(self.vol_sidelen, mask_rad)
            self.mask_input = Mask(self.sidelen_input, mask_rad)

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        idx: int

        Returns
        -------
        in_dict: Dictionary
        """
        particle = self.df['particles'].iloc[idx + self.idx_min]
        try:
            # Load particle image from mrcs file
            imgname_raw = particle['rlnImageName']
            imgnamedf = particle['rlnImageName'].split('@')
            mrc_path = os.path.join(self.path_to_starfile, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode='r', permissive=True) as mrc:
                proj = torch.from_numpy(mrc.data[pidx]).float() * self.scale_images
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)

            proj_input = tvf.resize(proj, [self.sidelen_input]*2)
            if self.mask_rad > 1e-3:
                proj_input = self.mask_input(proj_input)

            if self.resize_input is not None:
                proj = tvf.resize(proj, [self.vol_sidelen]*2)

            if self.mask_rad > 1e-3:
                proj = self.mask(proj)

        except Exception:
            print(f"WARNING: Particle image {particle['rlnImageName']} invalid!\nSetting to zeros.")
            proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj = proj[None, :, :]
            proj_input = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj_input = proj_input[None, :, :]

        # Generate CTF from CTF paramaters
        defocusU = torch.from_numpy(np.array(particle['rlnDefocusU'] / 1e4, ndmin=2)).float()
        defocusV = torch.from_numpy(np.array(particle['rlnDefocusV'] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()

        # Read "GT" orientations
        rotmat = torch.from_numpy(
            euler_angles2matrix(
                np.radians(-particle['rlnAngleRot']),
                np.radians(particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.radians(-particle['rlnAnglePsi'])
            )
        ).float()

        # Read "GT" shifts
        if self.no_trans:
            shiftX = torch.tensor([0.])
            shiftY = torch.tensor([0.])
        else:
            shiftX = torch.from_numpy(np.array(particle['rlnOriginXAngst']))
            shiftY = torch.from_numpy(np.array(particle['rlnOriginYAngst']))

        fproj = primal_to_fourier_2D(proj)

        in_dict = {'proj': proj,
                   'proj_input': proj_input,
                   'rotmat': rotmat,
                   'defocusU': defocusU,
                   'defocusV': defocusV,
                   'shiftX': shiftX,
                   'shiftY': shiftY,
                   'angleAstigmatism': angleAstigmatism,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   'fproj': fproj,
                   'imgname_raw': imgname_raw}

        return in_dict


class DensityMapProjectionSimulator(Dataset):
    def __init__(self, mrc_filepath, projection_sz, num_projs=None,
                 noise_generator=None, ctf_generator=None, power_signal=1,
                 resolution=3.2, shift_generator=None, fproj_mode='dft'):
        """
        Initialization of a dataloader from a mrc, simulating a cryo-EM experiment.

        Parameters
        ----------
        config: namespace
        """
        self.projection_sz = projection_sz
        self.num_projs = num_projs
        self.noise_generator = noise_generator
        self.ctf_generator = ctf_generator
        self.shift_generator = shift_generator
        self.fproj_mode = fproj_mode

        ''' Read mrc file '''
        self.mrc_filepath = mrc_filepath
        with mrcfile.open(mrc_filepath) as mrc:
            mrc_data = np.copy(mrc.data)
            power_init = get_power(mrc_data)
            mrc_data = 2e4 * power_signal * mrc_data * mrc_data.shape[0] / (power_init * self.projection_sz[0])
            voxel_size = float(mrc.voxel_size.x)
            if voxel_size < 1e-3:  # voxel_size = 0.
                voxel_size = resolution
                # voxel_size = 0.617
        self.mrc = mrc_data
        self.vol = torch.from_numpy(self.mrc).float()
        self.fvol_cpx = primal_to_fourier_3D(self.vol)  # S, S, S
        self.fvol = torch.cat([torch.real(self.fvol_cpx).unsqueeze(0),
                               torch.imag(self.fvol_cpx).unsqueeze(0)])  # 2, S, S, S

        x_lim = resolution * self.projection_sz[0] / (voxel_size * mrc_data.shape[0])

        ''' Planar coordinates '''
        lincoords = torch.linspace(-x_lim, x_lim, self.projection_sz[0])
        [X, Y] = torch.meshgrid([lincoords, lincoords])
        coords = torch.stack([Y, X, torch.zeros_like(X)], dim=-1)
        coords = shift_coords(coords, 1., 1., 0, self.projection_sz[0], self.projection_sz[0], 1)  # place DC component at (0, 0, 0)
        self.plane_coords = coords.reshape(-1, 3)

        ''' Volumetric coordinates '''
        self.vol_sidelen = self.projection_sz[0]
        self.vol_shape = [self.vol_sidelen] * 3
        lincoords = torch.linspace(-x_lim, x_lim, self.vol_sidelen)  # assume square volume
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        self.vol_coords = torch.stack([Y, X, Z], dim=-1).reshape(-1, 3)  # S^3, 3

        ''' Rotations '''
        self.rotmat = random_rotations(self.num_projs)

        # Keep precomputed projections to avoid recomputing them
        # and to get the same random realizations (for e.g. for noise)
        self.precomputed_projs = [None]*self.num_projs
        self.precomputed_fprojs = [None] * self.num_projs

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        rotmat = self.rotmat[idx, :]

        # If the projection has been precomputed already, use it
        if self.precomputed_projs[idx] is not None:
            proj = self.precomputed_projs[idx]
            fproj = self.precomputed_fprojs[idx]
        else:  # otherwise precompute it
            if self.fproj_mode == 'dft':
                rot_vol_coords = torch.matmul(self.vol_coords, rotmat)  # S^3, 3

                ''' Generate proj '''
                rot_vol = torch.nn.functional.grid_sample(self.vol[None, None, :, :, :],  # B, 1, S, S, S
                                                          rot_vol_coords[None, None, None, :, :],
                                                          align_corners=True)
                proj = torch.sum(rot_vol.reshape(self.vol_shape),  dim=-1)
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt) --> C, H, W

                ''' Generate fproj (fourier) '''
                fproj = primal_to_fourier_2D(proj)

            else:
                rot_plane_coords = torch.matmul(self.plane_coords, rotmat)  # S^2, 3
                rot_plane_coords = shift_coords(rot_plane_coords, 1., 1., 1., self.projection_sz[0],
                                                self.projection_sz[0],
                                                self.projection_sz[0],
                                                flip=True)

                ''' Generate fproj (fourier) '''
                fplane = torch.nn.functional.grid_sample(self.fvol[None, :, :, :, :],  # B, 2, S, S, S
                                                          rot_plane_coords[None, None, None, :, :],
                                                          align_corners=True)
                fplane = fplane.reshape(2, self.projection_sz[0], self.projection_sz[0]).permute(1, 2, 0)
                fplane = fplane.contiguous()
                fproj = torch.view_as_complex(fplane)[None, :, :]

                # ''' Generate proj '''
                # proj = torch.real(fourier_to_primal_2D(fproj))

            ''' CTF model (fourier) '''
            if self.ctf_generator is not None:
                fproj = self.ctf_generator(fproj, [idx])[0, ...]
                if hasattr(self.ctf_generator, 'variable_ctf'):
                    if self.ctf_generator.variable_ctf:
                        defocusU = self.ctf_generator.defocusU[idx]
                        defocusV = self.ctf_generator.defocusV[idx]
                        angleAstigmatism = self.ctf_generator.angleAstigmatism[idx]
                    else:
                        defocusU = self.ctf_generator.defocusU.reshape(1, 1)
                        defocusV = self.ctf_generator.defocusV.reshape(1, 1)
                        angleAstigmatism = self.ctf_generator.angleAstigmatism.reshape(1, 1)
                else:
                    defocusU = 0
                    defocusV = 0
                    angleAstigmatism = 0

            ''' Shift '''
            if self.shift_generator is not None:
                fproj = self.shift_generator(fproj, [idx])[0, ...]
                if hasattr(self.shift_generator, 'shifts'):
                    shiftX = self.shift_generator.shifts[idx, 0]
                    shiftY = self.shift_generator.shifts[idx, 1]
                else:
                    shiftX = 0.
                    shiftY = 0.

            ''' Update primal proj '''
            proj = fourier_to_primal_2D(fproj).real

            ''' Noise model (primal) '''
            if self.noise_generator is not None:
                proj = self.noise_generator(proj)

            ''' sync fproj with proj '''
            fproj = primal_to_fourier_2D(proj)

            ''' Store precomputed projs / fproj '''
            self.precomputed_projs[idx] = proj
            self.precomputed_fprojs[idx] = fproj

        in_dict = {'proj': proj,
                   'rotmat': rotmat,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   'fproj': fproj}
        in_dict['proj_input'] = proj

        if self.ctf_generator is not None:
            in_dict['defocusU'] = defocusU
            in_dict['defocusV'] = defocusV
            in_dict['angleAstigmatism'] = angleAstigmatism

        if self.shift_generator is not None:
            in_dict['shiftX'] = shiftX
            in_dict['shiftY'] = shiftY

        return in_dict