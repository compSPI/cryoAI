# CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images

This repository is the official implementation of [CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images](https://arxiv.org/abs/2203.08138)

![overview](https://user-images.githubusercontent.com/57400415/169631206-ae9e2166-066b-4f98-9642-acfd5fe8ab7b.png)

## Requirements

To install requirements in your current environment:

```setup
pip install -r requirements.txt
```

However, we recommend the container approach which guarantees more reproducible results over time. 
We provide methods to build a Docker image from a `Dockerfile` in the `docker/` sub-directory.
One could also directly "pull" the image from [Dockerhub](https://hub.docker.com/repository/docker/fpoitevi/cryoai):
```setup
docker pull fpoitevi/cryoai
```

For more information, please read [docker/README.md](https://github.com/compSPI/cryoAI/blob/main/docker/README.md).

## Running a Job

All jobs can be run with `src/reconstruct/main.py` and a config file (`.ini`). The config file must at least specify the `experiment_name` and `experiment_type`. To run a jub, use the following command:
```train
python -m src.reconstruct.main -c RELATVE_PATH_TO_CONFIG_FILE
```
The job can also be launched with a singularity container (`.sif`) using the command
```
singularity exec --nv ABSOLUTE_PATH_TO_CONTAINER python -m src.reconstruct.main -c RELATVE_PATH_TO_CONFIG_FILE
```
If using a shared computational resource managed with slurm, the job can be launched from the script `run_from_config.sh`
```slurm
sbatch run_from_config.sh -c RELATVE_PATH_TO_CONFIG_FILE --sif ABSOLUTE_PATH_TO_CONTAINER
```
Finally, if using a docker image you can launch a job with
```
...
```

## Generating a Synthetic Dataset

You must set `experiment_type = generate_synthetic` in the config file and specify the path to your mrc file with the `simul_mrc` argument. An example of config file is provided in `configfiles/mrc2star_80S_128.ini`. By default, generated starfiles are stored in the `simulated_starfiles/` directory.

## Loading a Real Dataset

Instructions to download datasets from EMPIAR car be found [here](https://github.com/zhonge/cryodrgn_empiar).

## Training

You must set `experiment_type = train` in the config file and specify the path to your starfile with the `path_to_starfile` argument and the name of the starfile with the `starfile` argument. An (incomplete) example is provided in `configfiles/train_80S_128.ini`. By default, log files containing tensorboard summaries, mrc files (reconstructed volumes) and config files are stored in the `logs/` directory. You can monitor your model with the following command:
```
tensorboard --logdir logs --port 8888 --bind_all
```

## Evaluation

Trained models are saved in a `.pth` format in the `logs/` directory. We give a pretrained model in `pretrained_models/80S_128/`. This model was trained on the 80S synthetic dataset generated with `mrc2star_80S_128.ini` (full config files for data generation and training are given in `pretrained_models/80S_128/`). You can use the encoder of this model to predict the poses on a full dataset by setting `experiment = evaluate_encoder` and specifying the path to the pth file with the `path_to_warm_start` argument. An example is provided in `configfiles/evaluate_80S_128.ini`.

## Results

A dataset was generated with `configfiles/mrc2star_80S_128.ini` and a cryoAI model was trained with `configfiles/train_80S_128.ini`.

![Screen Shot 2022-07-09 at 10 48 27 AM](https://user-images.githubusercontent.com/57400415/178117198-1eedb33a-d373-4295-b4eb-da596bfb26ce.png)
