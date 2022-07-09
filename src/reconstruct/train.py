"""Experiment for training a cryoAI model."""

from torch.utils.data import DataLoader

from ..cryoai import CryoAI
from ..dataio import StarfileDataLoader
from ..training_chunks import train
from ..loss_utils import loss_factory


def experiment(config):
    """
    Experiment for training a cryoAI model.

    Parameters
    ----------
    config: namespace
    """
    # Dataset
    dataset = StarfileDataLoader(config)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=config.train_batch_sz, pin_memory=True,
                            num_workers=config.train_num_workers, drop_last=True)

    # Model
    cryoai = CryoAI(config)
    cryoai.cuda()

    # Warm start
    if config.warm_start_full:
        cryoai.warm_start(config.path_to_warm_start)
    elif config.warm_start_volume:
        cryoai.warm_start_volume(config.path_to_warm_start)

    # Losses
    loss_dict = loss_factory(config)

    # Loss schedules
    loss_schedules = {'data_term': lambda e, t: 1,
                      'contrastive_term': lambda e, t: config.contrastive_loss_weight *
                                                       float(e < config.contrastive_epochs),
                      'tv_fourier_term': lambda e, t: config.TV_fourier_reg_weight,
                      'tv_primal_term': lambda e, t: config.TV_primal_reg_weight}

    # Training
    train(model=cryoai,
          train_dataloader=dataloader,
          num_particles=config.num_particles,
          epochs=config.train_epochs,
          optimizer=config.optimizer,
          lr=config.train_learning_rate,
          lr_encoder_scaling=config.lr_encoder_scaling,
          steps_til_light_summary=config.steps_til_light_summary,
          epochs_til_heavy_summary=config.epochs_til_heavy_summary,
          root_dir=config.root_dir,
          model_dir=config.model_dir,
          loss_dict=loss_dict,
          loss_schedules=loss_schedules,
          max_chunk_sz=config.train_chunk_sz,
          fast_mode=config.fast_mode,
          write_mrc=config.write_mrc,
          flip_images=config.flip_images,
          print_times=config.print_times
          )
