"""main python file"""

import sys
import os

import traceback
import configargparse

from ..utils import cond_mkdir
from .train import experiment as train
from .evaluate_encoder import experiment as evaluate_encoder


def init_config(parser):
    # Resources
    parser.add_argument('--gpu', type=int, default=0,
                        help='The GPU instance to select for training.')
    parser.add_argument('--job_id', type=str, help='Job id in slurm.')
    parser.add_argument('--train_num_workers', type=int, default=8,
                        help='The number of workers to use for the dataloader. (>0 not implemented)')
    parser.add_argument('--train_batch_sz', type=int, default=256,
                        help='The number of projections in a batch for training.')
    parser.add_argument('--train_chunk_sz', type=int, default=256,
                        help='The size of a chunk of views that can fit on the GPU,'
                             'chunk_sz < batch_size. The batch will be automatically divided'
                             'in chunks of chunk_sz and gradients are accumulated over the chunk'
                             'so that we perform one SGD step every batch_sz.')
    # Optimization
    parser.add_argument('--train_learning_rate', type=float, default=1e-4,
                        help='The learning rate used during training.')
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='The number of epochs of training.')
    parser.add_argument('--lr_encoder_scaling', type=float, default=1e0,
                        help='The learning rate of the encoder is multiplied by lr_encoder_scaling.')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam',
                        help='Type of optimizer.')
    # Summary / Checkpoints
    parser.add_argument('--steps_til_light_summary', type=int, default=500,
                        help='The number of steps (in #batches) between light summaries.')
    parser.add_argument('--epochs_til_heavy_summary', type=int, default=1,
                        help='The number of epochs between heavy summaries.')
    parser.add_argument('--print_times', type=int, default=0,
                        help='Print computation times.')
    parser.add_argument('--compute_proj', type=int, default=1,
                        help='Compute the ifft of fproj in cryoai?')
    parser.add_argument('--write_mrc', type=int, default=1,
                        help='Write mrc file?')
    parser.add_argument('--fast_mode', type=int, default=0,
                        help="Deactivates summaries to decrease computation time.")
    # Data Loading
    parser.add_argument('--side_len', type=int, default=64,
                        help='The shape of the density map as determined by one side of the volume.')
    parser.add_argument('--side_len_input_output_ratio', type=int, default=1,
                        help='Ratio between input / output side lengths.')
    parser.add_argument('--num_particles', type=int, default=10000,
                        help='Total number of particle images to use in reconstruction.')
    parser.add_argument('--path_to_starfile', type=str,
                        help='The root directory for a RELION run.')
    parser.add_argument('--starfile', type=str,
                        help='The filepath to RELION\'s star file.')
    parser.add_argument('--invert_hand', type=bool, default=0,
                        help='Invert handedness when reading relion data.')
    parser.add_argument('--mask_rad', type=float, default=0.,
                        help='Mask radius between 0 and 1, 0 for no mask.')
    parser.add_argument('--idx_min', type=int, default=0,
                        help='Minimal index in the relion dataset.')
    parser.add_argument('--idx_max', type=int, default=-1,
                        help='Maximal index in the relion dataset.')
    parser.add_argument('--scale_images', type=float, default=1.,
                        help='Scaling of images.')
    parser.add_argument('--no_trans', type=int, default=0,
                        help='No translation in the dataset.')
    # Encoder
    parser.add_argument('--encoder_batch_norm', type=int, default=1,
                        help='Use batch-norm in the CNN encoder?')
    parser.add_argument('--gaussian_pyramid', type=int, default=1,
                        help='Use a gaussian pyramid of filters before the encoder?')
    parser.add_argument('--num_octaves', type=int, default=4,
                        help='Number of octaves in the gaussian pyramid.')
    parser.add_argument('--encoder_type', type=str, choices=['vgg16'], default='vgg16',
                        help="Type of encoder.")
    # Poses
    parser.add_argument('--regressor_orientation_layers', type=list, default=[512, 256],
                        help="Number of features in the FC layers of the orientation regressor.")
    parser.add_argument('--regressor_shift_layers', type=list, default=[512, 256],  # [1024, 512, 256]
                        help="Number of features in the FC layers of the shift regressor.")
    parser.add_argument('--so3_parameterization', type=str, default='s2s2',
                        choices=['s2s2', 'euler', 'quaternion', 'gt'],
                        help='The parameterization of SO3 influences the interpretation of the output of the'
                             ' orientation regressor.')
    parser.add_argument('--pose_estimation', type=str, default='encoder', choices=['encoder', 'gt'],
                        help="Estimation of the poses using an encoder of an autodecoder (static-learnt)")
    # Volume representation
    parser.add_argument('--volume_representation', type=str, choices=['imp-fourier'],
                        default='imp-fourier',
                        help='Volume representation.')
    parser.add_argument('--implicit_representation_type', type=str, choices=['siren', 'relu_pe', 'fouriernet'],
                        default='fouriernet',
                        help='Type of implicit representation.')
    parser.add_argument('--force_symmetry', type=int, default=0,
                        help='Forces the FourierNet to give an hermitian function.')
    # CTF
    parser.add_argument('--use_ctf', type=str, choices=['gt'], default='gt',
                        help="gt: use ctf from gt. static: use the same ctf for each particle. "
                             "none: identity ctf.")
    parser.add_argument('--ctf_size', type=int, default=128,
                        help='The size of the CTF filter used in reconstructions.')
    parser.add_argument('--ctf_valueNyquist', type=float, default=0.001,
                        help='Reconstruction CTF value at Nyquist.')
    parser.add_argument('--ctf_precompute', type=int, default=0,
                        help='Precompute CTF filters (set False for large datasets). Default = False')
    parser.add_argument('--kV', type=float, default=300.0,
                        help='Electron beam energy used.')
    parser.add_argument('--resolution', type=float, default=.8,
                        help='Particle image resolution (in Angstrom).')
    parser.add_argument('--spherical_abberation', type=float, default=2.7,
                        help='Spherical aberration.')
    parser.add_argument('--amplitude_contrast', type=float, default=0.1,
                        help='Amplitude contrast.')
    parser.add_argument('--ctf_device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Which device should the parameters of the CTFs (for the model) be stored?')
    # Shift
    parser.add_argument('--use_shift', type=str, choices=['gt', 'encoder'], default='encoder',
                        help='Whether to use the shift in the slices and whether gt is provided.')
    # Loss
    parser.add_argument('--data_loss_domain', type=str, default='primal', choices=['primal', 'fourier'],
                        help='In which domain should the data loss operate?')
    parser.add_argument('--data_loss_norm', type=str, default='symloss',
                        choices=['L1', 'L2', 'symloss'],
                        help='Norm for data loss.')
    parser.add_argument('--use_contrastive_loss', type=int, default=0,
                        help='Use contrastive loss?')
    parser.add_argument('--contrastive_epochs', type=int, default=-1,
                        help='The number epochs to apply the contrastive loss.')
    parser.add_argument('--contrastive_loss_weight', type=str, default=1.,
                        help='Weight of contrastive loss.')
    parser.add_argument('--use_masked_loss', type=int, default=0,
                        help='Use circular mask on the loss?')
    parser.add_argument('--mask_rad_loss', type=float, default=0.75,
                        help='Mask radius in loss.')
    # Frequency Marcher
    parser.add_argument('--use_frequency_marcher', type=int, default=0,
                        help='Use the frequency marcher?')
    parser.add_argument('--test_convergence_every', type=int, default=250,
                        help='Number of steps between two tests of convergence.')
    # Warm Start
    parser.add_argument('--warm_start_volume', type=int, default=0,
                        help='Make a warm start on the decoder?')
    parser.add_argument('--warm_start_full', type=int, default=0,
                        help='Make a warm start on the encoder and the decoder?')
    parser.add_argument('--path_to_warm_start', default=None,
                        help='Path to .pth file for warm start')
    # Output Starfile
    parser.add_argument('--model_output_starfile', type=str, default='encoder',
                        help="Model used to predict orientations and write starfile.")


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True,
                        help='Path to config file.')

    # We must give an experiment name
    parser.add_argument('--experiment_type', type=str, default='exp_simul_proj',
                        choices=['train',
                                 'evaluate_encoder'],
                        help='The experiment to run.')
    parser.add_argument('--experiment_name', type=str, default=None, required=True,
                        help='An identifier for the train run.')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Output directory for logging.')
    parser.add_argument('--output_starfile_dir', type=str, default='output_starfiles_v1/',
                        help='Output directory for starfiles after reconstruction.')

    # This function will initialize a config file if it does not already exist,
    # by filling it with default parameters
    init_config(parser)
    config = parser.parse_args()
    config.map_shape = [config.side_len] * 3

    if config.experiment_name is None:
        parser.error('Error: --experiment_name is required.')
    if config.job_id is not None:
        config.experiment_name = config.experiment_name + '_' + config.job_id

    # Assert that the config options are self-consistent
    if config.warm_start_volume or config.warm_start_full:
        assert config.path_to_warm_start is not None, "path_to_warm_start cannot be None."
    if config.use_frequency_marcher:
        assert config.volume_representation == 'imp-fourier', "Frequency marcher can only be activated for" \
                                                              " imp-fourier representation."
    if config.encoder_type == 'e2cnn':
        assert config.so3_parameterization in ['euler'], "e2cnn encoder can only be used with euler" \
                                                         " parameterization of SO(3)"
    if config.data_loss_norm == 'symloss':
        config.flip_images = 1
    if config.mask_rad > 1e-3:
        assert config.mask_rad >= config.mask_rad_loss, "The mask on the loss must be smaller than" \
                                                        " the mask on the input images"

    # Create root directory where models, logs and config files will be written
    config.root_dir = os.path.join(config.log_dir, config.experiment_name)
    if not cond_mkdir(config.root_dir):
        print(f"Error: cannot create root path.")
        return -1

    config.model_dir = os.path.join(config.root_dir, 'models')
    cond_mkdir(config.model_dir)

    # Write the config file
    parser.write_config_file(config, [os.path.join(config.root_dir, 'config.ini')])

    # Select the local gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    # Finally: launch the experiment!
    print(f"Launching experiment {config.experiment_type}")
    if config.experiment_type == 'train':
        train(config)
    elif config.experiment_type == 'evaluate_encoder':
        evaluate_encoder(config)

    return 0, 'Training successful'


if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Training failed.'

    print(status_message)
    exit(retval)
