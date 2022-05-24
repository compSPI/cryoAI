# CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images

This repository is the official implementation of [CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images](https://arxiv.org/abs/2203.08138)

![overview](https://user-images.githubusercontent.com/57400415/169631206-ae9e2166-066b-4f98-9642-acfd5fe8ab7b.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Data

...

## Homogeneous Reconstruction

Configuration parameters must be save in a `.ini` file. We provide an example in `configfiles/spliceosome.ini`.

To train a cryoAI model, run this command:

```train
python -m src.reconstruct.main -c RELATVE_PATH_TO_CONFIG_FILE
```

By default, log files containing tensorboard summaries and `.mrc` files (reconstructed volumes) are stored in the `logs` directory.

If using a shared computational resource managed with slurm, the job can be launched with a singularity container (`.sif`) using the command:

```slurm
sbatch run_from_config.sh -c RELATVE_PATH_TO_CONFIG_FILE --sif ABSOLUTE_PATH_TO_CONTAINER
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
