import torch
import torch.fft
import mrcfile
import starfile
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
import numpy as np
import os
from .geom_utils import euler_angles2matrix
from .ctf_utils import primal_to_fourier_2D
from .mask_utils import Mask


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
                   'fproj': fproj}

        return in_dict
