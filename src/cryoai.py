"""cryoAI class"""

import torch
import torch.fft
from torch import nn
from .ml_modules import FCBlock, CNNEncoderVGG16
from .ctf_utils import fourier_to_primal_2D, CTFRelion
from .shift_utils import Shift, ShiftIdentity
from .volume_utils import ImplicitFourierVolume
from .filter_utils import GaussianPyramid
from .scheduler_utils import FrequencyMarcher
from .geom_utils import add_flipped_rotmat
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, quaternion_to_matrix
import os
import time


class CryoAI(nn.Module):
    def __init__(self, config):
        """
        Initialization of a cryoAI model.

        Parameters
        ----------
        config: namespace
        """
        super(CryoAI, self).__init__()
        self.config = config
        self.sidelen_input = config.side_len * config.side_len_input_output_ratio

        # Schedulers
        self.schedulers = {}
        if config.use_frequency_marcher:
            self.frequency_marcher = FrequencyMarcher(self, config.side_len, config.test_convergence_every)
            self.schedulers['frequency_marcher'] = self.frequency_marcher
        else:
            self.frequency_marcher = None

        # Volume representation
        if config.volume_representation == 'imp-fourier':
            params_implicit = {"type": config.implicit_representation_type, "force_symmetry": config.force_symmetry}
            self.pred_map = ImplicitFourierVolume(
                img_sz=config.map_shape[0],
                params_implicit=params_implicit,
                frequency_marcher=self.frequency_marcher
            )
        else:
            raise NotImplementedError

        # Pre-filtering
        self.gaussian_pyramid = config.gaussian_pyramid
        if config.gaussian_pyramid:
            self.gaussian_filters = GaussianPyramid(
                kernel_size=11,
                kernel_variance=0.01,
                num_octaves=config.num_octaves,
                octave_scaling=10
            )
            num_additional_channels = config.num_octaves
        else:
            num_additional_channels = 0

        # Encoder
        if config.encoder_type == 'vgg16':
            self.cnn_encoder = CNNEncoderVGG16(
                in_channels=1+num_additional_channels,
                batch_norm=config.encoder_batch_norm,
                pretrained=False,
                flip_images=config.flip_images,
                high_res=(self.sidelen_input > 128)
            )
            map_height = self.sidelen_input
            map_width = self.sidelen_input
            cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(map_height, map_width)
            latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))
        else:
            raise NotImplementedError

        # Orientation regressors
        if self.config.so3_parameterization == 'euler':
            self.orientation_dims = 3
            self.equiv_dims = 1
            self.invar_dims = 2
            self.last_nonlinearity = None
            self.latent_to_rot3d_fn = lambda x: euler_angles_to_matrix(x, 'ZYZ')
        elif self.config.so3_parameterization == 'quaternion':
            self.orientation_dims = 4
            self.last_nonlinearity = None
            self.latent_to_rot3d_fn = lambda x: quaternion_to_matrix(x / torch.norm(x, p=2, dim=-1, keepdim=True))
        elif self.config.so3_parameterization == 's2s2':
            self.orientation_dims = 6
            self.equiv_dims = 3
            self.invar_dims = 3
            self.last_nonlinearity = None
            self.latent_to_rot3d_fn = rotation_6d_to_matrix
        elif self.config.so3_parameterization == 'gt':
            self.latent_to_rot3d_fn = None

        if config.pose_estimation == 'encoder':
            if self.latent_to_rot3d_fn is not None:
                # We split the regressor in 2 to have access to the latent code
                self.orientation_encoder = FCBlock(
                    in_features=latent_code_size,
                    out_features=config.regressor_orientation_layers[-1],
                    features=config.regressor_orientation_layers[:-1],
                    nonlinearity='relu',
                    last_nonlinearity='relu',
                    batch_norm=config.encoder_batch_norm
                )
                self.orientation_regressor = FCBlock(
                    in_features=config.regressor_orientation_layers[-1],
                    out_features=self.orientation_dims,
                    features=[],
                    nonlinearity='relu',
                    last_nonlinearity=self.last_nonlinearity,
                    batch_norm=config.encoder_batch_norm
                )

        # Shift regressor
        if config.use_shift == 'encoder':
            self.shift_encoder = FCBlock(
                in_features=latent_code_size,
                out_features=2,
                features=config.regressor_shift_layers,
                nonlinearity='relu',
                last_nonlinearity=None
            )

        # CTF model
        if config.use_ctf == 'gt':
            self.ctf = CTFRelion(size=config.ctf_size, resolution=config.resolution,
                                 kV=config.kV, valueNyquist=config.ctf_valueNyquist, cs=config.spherical_abberation,
                                 amplitudeContrast=config.amplitude_contrast, requires_grad=False,
                                 num_particles=config.num_particles, precompute=config.ctf_precompute,
                                 flip_images=config.flip_images)

        # Shift model
        if config.use_shift in ['encoder', 'gt']:
            self.shift = Shift(
                num_particles=config.num_particles,
                size=config.ctf_size,
                resolution=config.resolution,
                requires_grad=False,
                flip_images=config.flip_images
            )
        else:
            self.shift = ShiftIdentity()

    def warm_start(self, path_to_warm_start):
        """
        Warm start encoder + decoder.

        Parameters
        ----------
        path_to_warm_start: str
        """
        print("Full warm start.")
        self.load_state_dict(torch.load(path_to_warm_start))

    def warm_start_volume(self, path_to_warm_start_volume):
        """
        Warm start decoder.

        Parameters
        ----------
        path_to_warm_start_volume: str
        """
        print("Warm start volume.")
        self.pred_map.load_state_dict(torch.load(path_to_warm_start_volume))

    def save(self, checkpoints_dir, epoch):
        """
        Saves the model.

        Parameters
        ----------
        checkpoints_dir: abspath
        epoch: int
        """
        torch.save(self.state_dict(),
                   os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
        torch.save(self.pred_map.state_dict(),
                   os.path.join(checkpoints_dir, 'pred_map_epoch_%04d.pth' % epoch))

    def forward(self, in_dict):
        """
        Forward pass.

        Parameters
        ----------
        in_dict: dictionary

        Returns
        -------
        output_dict: dictionary
        """
        start_time = time.time()
        proj = in_dict['proj_input']

        # Schedulers
        for name, scheduler in self.schedulers.items():
            scheduler.update()

        latent_code = None

        # Poses
        if self.config.so3_parameterization == 'gt':
            pred_rotmat = in_dict['rotmat']
            latent_code_prerot = None
            if self.config.flip_images:
                pred_rotmat = add_flipped_rotmat(pred_rotmat)
        else:
            if self.config.pose_estimation == 'encoder':
                if self.gaussian_pyramid:
                    proj = self.gaussian_filters(proj)

                latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
                latent_code_pose = self.orientation_encoder(latent_code)
                latent_code_prerot = self.orientation_regressor(latent_code_pose)
            else:
                latent_code_prerot = self.poses(in_dict['idx'])

            # Interpret the latent code as a rotation
            pred_rotmat = self.latent_to_rot3d_fn(latent_code_prerot)

        encoder_time = time.time()

        # Shift
        if self.config.use_shift == 'gt':
            pred_shift_params = {k: in_dict[k] for k in ('shiftX', 'shiftY')
                            if k in in_dict}
        elif self.config.use_shift == 'encoder':
            if self.gaussian_pyramid:
                latent_code = torch.flatten(self.cnn_encoder(self.gaussian_filters(proj)), start_dim=1)\
                    if latent_code is None else latent_code
            else:
                latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1) if latent_code is None else latent_code
            shift_params = self.shift_encoder(latent_code)
            pred_shift_params = {'shiftX': shift_params[..., 0].reshape(-1),
                                 'shiftY': shift_params[..., 1].reshape(-1)}
        else:
            raise NotImplementedError

        # CTF
        if self.config.use_ctf == 'gt':
            pred_ctf_params = {k: in_dict[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism')
                               if k in in_dict}
        else:
            raise NotImplementedError

        ctf_shift_prep_time = time.time()

        # Query the volume (slicing / projection)
        pred_fproj_prectf = self.pred_map(pred_rotmat)

        decoder_time = time.time()

        # Apply the remaining step of the forward model: CTF + Shift
        pred_fproj = self.ctf(
            pred_fproj_prectf,
            in_dict['idx'],
            pred_ctf_params,
            mode=self.config.use_ctf,
            frequency_marcher=self.frequency_marcher
        )
        pred_fproj = self.shift(
            pred_fproj,
            in_dict['idx'],
            pred_shift_params,
            mode=self.config.use_shift,
            frequency_marcher=self.frequency_marcher
        )

        end_time = time.time()

        times = {'start': start_time,
                 'encoder': encoder_time,
                 'ctf_shift_prep': ctf_shift_prep_time,
                 'decoder': decoder_time,
                 'end': end_time}

        # Fill the output dictionary
        output_dict = {'rotmat': pred_rotmat,
                       'latent_code': latent_code,
                       'latent_code_prerot': latent_code_prerot,
                       'fproj': pred_fproj,
                       'fproj_prectf': pred_fproj_prectf,
                       'pred_ctf_params': pred_ctf_params,
                       'pred_shift_params': pred_shift_params,
                       'times': times}

        # Make sure we are in sync by bringing back the proj to the primal domain
        if self.config.data_loss_domain == 'primal' or self.config.compute_proj:
            pred_proj = fourier_to_primal_2D(pred_fproj)
            output_dict['proj'] = pred_proj.real

        fproj_gt = in_dict['fproj']  # B, 1, S, S
        if self.frequency_marcher is not None:
            img_sz = self.config.map_shape[0]
            left = max(0, img_sz // 2 - self.frequency_marcher.f)
            right = min(img_sz, img_sz // 2 + self.frequency_marcher.f + 1)
            fproj_gt = fproj_gt[:, :, left:right, left:right]  # B, 1, 2*f+1, 2*f+1
        output_dict['fproj_gt'] = fproj_gt
        if self.config.data_loss_domain == 'primal' or self.config.compute_proj:
            proj_gt = fourier_to_primal_2D(fproj_gt)
            output_dict['proj_gt'] = proj_gt.real

        return output_dict
