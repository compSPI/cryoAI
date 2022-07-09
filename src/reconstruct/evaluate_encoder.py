"""Experiment for pose prediction with a pre-trained cryoAI encoder."""

from torch.utils.data import DataLoader

from ..cryoai import CryoAI
from ..dataio import StarfileDataLoader
from ..training_chunks import eval_to_starfile
from ..ml_modules import OrientationPredictor
import os
from ..utils import cond_mkdir


def experiment(config):
    dataset = StarfileDataLoader(config)
    dataloader = DataLoader(dataset,
                            shuffle=True, batch_size=config.train_batch_sz,
                            pin_memory=True, num_workers=config.train_num_workers,
                            drop_last=True)

    cryoai = CryoAI(config)

    ''' Warm Starts '''
    if config.warm_start_full:
        cryoai.warm_start(config.path_to_warm_start)

    orientation_predictor = OrientationPredictor(cryoai)
    orientation_predictor.cuda()
    orientation_predictor.eval()
    model = orientation_predictor

    root_dir = os.path.join(config.output_starfile_dir, config.experiment_name)
    if not cond_mkdir(root_dir):
        print(f"Error: cannot create root path.")
        return -1

    print("Creating starfile")
    eval_to_starfile(model=model,
                     dataloader=dataloader,
                     epochs=config.train_epochs,
                     root_dir=root_dir,
                     name=config.experiment_name,
                     config=config)

