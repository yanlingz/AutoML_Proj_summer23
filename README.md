# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project

This repository contains all things needed for the final projects.
This project aims to optimize a CNN architecture using multi-fidelity optimization.

### (Recommended) Setup new clean environment

Use a package manager, such as the one provided by your editor, python's built in `venv`
or [miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements).

#### Conda
Subsequently, *for example*, run these commands, following the prompted runtime instructions:
```bash
conda create -n automl python=3.10
conda activate automl
pip install -r requirements.txt
```

#### Venv

```bash
# Make sure you have python 3.8/3.9/3.10
python -V
python -m venv my-virtual-env
./my-virtual-env/bin/activate
pip install -r requirements.txt
```

#### SMAC
If you have issues installing SMAC,
follow the instructions [here](https://automl.github.io/SMAC3/main/1_installation.html).


### Data
You need to pre-download all the data required by running `python datasets.py`.

Stores by default in a `./data` directory. Takes under 20 seconds to download and extract.

#### `multi_fidelity_search.py`
* Uses SMAC for multi-fidelity optimization.
* Uses `--scenario` name to obtain information about approach and adapt configspace and model to be optimized
* Evaluates final incumbent and writes result to output directory

#### `cellspace_search.py`
* Implements a network for CellSpace search
* Implements DARTS Normal and Reduction Cell Architectures
* NormalCell -> ReductionCell -> NormalCell -> ReductionCell -> FC
* Works with SMAC to obtain the best operation configurations

#### `eval_incumbent.py`
* Script for testing the configurations on test dataset

#### `configspaces`
* This directory contains the various configspaces used for experiments

