# CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images

This repository is the official implementation of [CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images](https://arxiv.org/abs/2203.08138)

![overview](https://user-images.githubusercontent.com/57400415/169631206-ae9e2166-066b-4f98-9642-acfd5fe8ab7b.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Running a Job

All jobs can be run with 'src/reconstruct/main.py' and a config file (`.ini`):
```train
python -m src.reconstruct.main -c RELATVE_PATH_TO_CONFIG_FILE
```
The config file must at least specify the `experiment_name` and `experiment_type`.

If using a shared computational resource managed with slurm, the job can be launched with a singularity container (`.sif`) using the command
```slurm
sbatch run_from_config.sh -c RELATVE_PATH_TO_CONFIG_FILE --sif ABSOLUTE_PATH_TO_CONTAINER
```

## Generating a Synthetic Dataset

You must set `experiment_type = generate_synthetic` in the config file and specify the path to your `.mrc` file with the `simul_mrc` argument. An example is provided in `configfiles/mrc2star_80S_128.ini`. Generate a synthetic dataset by running
```
python -m src.reconstruct.main -c configfiles/mrc2star_80S_128.ini
```

If using slurm,
```slurm
sbatch run_from_config.sh -c configfiles/mrc2star_80S_128.ini --sif ABSOLUTE_PATH_TO_CONTAINER
```

By default, starfiles are stored in the `simulated_starfiles` directory.

## Loading a Real Dataset

...

## Training

You must set `experiment_type = train` in the config file and specify the path to your `.star` file with the `path_to_starfile` argument and the name of the starfile with the `starfile` argument. An (incomplete) example is provided in `configfiles/spliceosome.ini`. To launch the training, run
```
python -m src.reconstruct.main -c configfiles/spliceosome.ini
```

If using slurm,
```slurm
sbatch run_from_config.sh -c configfiles/spliceosome.ini --sif ABSOLUTE_PATH_TO_CONTAINER
```

By default, log files containing tensorboard summaries and `.mrc` files (reconstructed volumes) are stored in the `logs` directory.

## Evaluation

...

## Results

...

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
