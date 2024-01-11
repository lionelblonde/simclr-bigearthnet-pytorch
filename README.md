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

GDAL: might need to install `gdal` and/or `libgdal` on your system.
Has proven finicky to install.

Python version: >=3.10

Set up your Python environment as follows (ordering is important):
```bash
pip install --upgrade pip
conda install -c conda-forge gdal
pip install rasterio opencv-python tqdm numpy scikit-learn wandb tmuxp tabulate pyright ruff-lsp
pip install torch torchvision torchaudio
```

