"""Experiment for generating a synthetic dataset from a mrc file."""

from torch.utils.data import DataLoader

from ..dataio import DensityMapProjectionSimulator
from ..ctf_utils import CTFIdentity, CTFRelion
from ..noise_utils import AWGNGenerator
from ..shift_utils import Shift, ShiftIdentity
from ..starfile_utils import create_starfile
from ..utils import cond_mkdir
import os
import torch.multiprocessing


def experiment(config):
    """ Creates a starfile from a mrc by simulating a cryo-EM experiment.

    Parameters
    ----------
    config - parser.
    """
    torch.multiprocessing.set_sharing_strategy('file_system')  # avoid Too many open files error

    print("Experiment Name: " + str(config.experiment_name))

    ''' Noise '''
    print("Creating simulation noise")
    noise = AWGNGenerator(use_noise=config.use_noise,
                          snr=config.snr,
                          power_signal=config.power_signal)

    ''' CTF '''
    print("Creating simulation CTF")
    if config.use_ctf != 'none':
        ctf = CTFRelion(size=config.map_shape[0], resolution=config.resolution,
                        defocusU=config.sim_ctf_defocus_u, defocusV=config.sim_ctf_defocus_v,
                        sim_ctf_defocus_stdev = config.sim_ctf_defocus_stdev,
                        angleAstigmatism=config.sim_ctf_angle_astigmatism,
                        cs=config.sim_ctf_spherical_abberations,
                        num_particles=config.simul_num_projs,
                        requires_grad=False, device='cuda',
                        variable_ctf=config.variable_ctf,
                        precompute=config.ctf_sim_precompute)
    else:
        ctf = CTFIdentity()
    # We use the parameters of the simulation here, to match those of the model.
    config.ctf_size = config.map_shape[0]
    config.spherical_aberration = config.sim_ctf_spherical_abberations
    config.num_particles = config.simul_num_projs

    ''' Shift '''
    print("Creating simulation shift")
    if config.use_shift != 'none':
        shift = Shift(num_particles=config.simul_num_projs,
                      size=config.map_shape[0],
                      resolution=config.resolution,
                      requires_grad=False,
                      std_shift=config.std_shift)
    else:
        shift = ShiftIdentity()

    ''' Dataloaders '''
    print("Creating simulation dataset")
    dataset = DensityMapProjectionSimulator(mrc_filepath=config.simul_mrc,
                                            projection_sz=config.map_shape,
                                            num_projs=config.simul_num_projs,
                                            noise_generator=noise,
                                            ctf_generator=ctf,
                                            power_signal=config.power_signal,
                                            resolution=config.resolution,
                                            shift_generator=shift)
    dataloader = DataLoader(dataset,
                            shuffle=False, batch_size=config.train_batch_sz,
                            pin_memory=False, num_workers=config.train_num_workers)

    root_dir = os.path.join(config.sim_starfile_dir, config.experiment_name)
    if not cond_mkdir(root_dir):
        print(f"Error: cannot create root path.")
        return -1

    relative_mrcs_path_prefix = 'Particles/'
    mrcs_dir = os.path.join(root_dir, relative_mrcs_path_prefix)
    if not cond_mkdir(mrcs_dir):
        print(f"Error: cannot create mrcs path.")
        return -1

    print("Creating starfile")
    create_starfile(dataloader,
                    config,
                    root_dir,
                    relative_mrcs_path_prefix,
                    config.experiment_name)