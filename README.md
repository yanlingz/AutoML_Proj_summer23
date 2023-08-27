# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project

This repository contains all things needed for the final projects.
The goal is to improve and analyze performance (wrt accuracy) of the CNN by AutoML means.
Framework is Multifidelity Optimization, using Hyperband Facade in SMAC3 (similar to BOHB algorithm).
This work has been done by team auoml.

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



#### Running
Train a model to use SMAC with multi-fidelity optimization to discover the best configuration that achieves optimal accuracy.
```bash
python multi_fidelity.py
```
* The configsapce that we used to get the best performance is in the file.
* JSON files contain partially experimented configspaces. 

#### Evaluation 

Evaluation is performed using the script CFG_evaluate.py. The currently best-discovered configuration has been included in the dictionary. 

```bash
# Make sure you have python 3.8/3.9/3.10
python CFG_evaluate.py 
```
