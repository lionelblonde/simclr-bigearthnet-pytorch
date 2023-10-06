# SimCLR in PyTorch

## Description

The repository contains PyTorch implementation of the SimCLR self-supervised learning method.

It is adapted to be used on earth observation data
from the BigEarthNet dataset/benchmark,
with __multilabel__ classification as downstream task
(whether with a linear probe or finetuning). 
The repository also offers an implementation of a classifier
for the same multilabel classification task as a standalone task,
so that a direct comparison with the outcome of SimCLR's downstream
task is easily and directly obtainable.


There is one script to run the job locally (`anton_launcher`)
and one adapted for Slurm-orchestrated HPC clusters (`slurm_launcher`).
The file `spawner.py` at the root (called by `slurm_launcher`)
also offer the option to spawn arrays of jobs locally within a Tmux session
(where each job spawned has its own window within the session).

When run locally, the scripts expect the dataset to be present locally _uncompressed_.
When run with Slurm, it expects the dataset to be present on a accessible note _compressed_.
These behaviors are modifiable in the scripts provided.
The choices I made here were for my own convenience.

## Requirements

Python >=3.10

Set up your Python environment as follows (prefered way; Dockerfile provided too):
```bash
conda install -c conda-forge gdal
pip install --upgrade pip
pip install rasterio
pip install opencv-python
pip install tqdm
pip install numpy
pip install scikit-learn
pip install wandb
pip install tmuxp
pip install tabulate
pip install torch torchvision torchaudio
```
