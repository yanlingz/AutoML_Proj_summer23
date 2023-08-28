# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project

Welcome to the repository for the final project of the AutoML lecture held at the University of Freiburg in Summer 2023.

`Objective`: The primary objective was to find a high-performing CNN within a constrained timeframe (6 hours) using AutoML methods. Specifically, the aim was to determine the best neural network architecture from a predefined configuration space, where each configuration stands for a unique architecture.
For clarity, consider the following exemplary configuration space:

Number of Layers: [2, 3, 4]

Neurons per Layer: [16, 32, 64, 128]

Initial Learning Rate: [0.1, 0.01, 0.0001]

A single config could be a neural network with 3 Layers, 64 neurons per layer and an initial learning rate of 0.001.
This is a simplified example for understanding purposes; the actual configuration space provided in the assignment was more extensive.

## Approach
`Data Augmentation` is a widely adopted strategy in Image Classification to improve the performance of a CNN. However, adding data augmentation to the existing configspace would expand the set of hyperparameters exponentially. For example, adding the possiblity of randomly cropping images during training, already doubles the amount of possible combinations. 

Thus, I have decided to split up the training process:

`Pre-Run`: Using the given configuration space / set of hyperparameters without any data augmentation, I utilized SMAC to swiftly identify an efficient configuration, limiting the training to just 3 epochs.

`Post-Run`: The best-performing CNN derived from the pre-run was trained on the maximum allowed number of epochs (20 epochs). This phase introduced 
 various data augmentation combinations and dropout rates. To optimize the training and evaluation time, I commenced with an image size of 8, scaling up to 32 in the later stages for BOHB.


## Prerequisites
In order to be able to run the training script, Python 3.10 or higher is required. To install the required packages run

```
pip install -r requirements.txt
```

Important Notes:

`Configspace Version`: I used an older version of Configspace, which was not used in the original repository of the project, due to an error I got on newer versions when trying to change the hyperparameters of a config.

`PyTorch Version`: This project incorporates features like CutMix and MixUp, which are available only in the nightly version of PyTorch (as of 29.08.2023). It's essential to use this most recent PyTorch version. Please ensure that your system is compatible with CUDA 12.1, I did not test it with CUDA 11.8. Also, please be aware that nighlty builds might contain experimental or unstable features.

`For Conda users:`

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```

`if you are using pip:`

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```


## Training

To run the full training pipeline with default parameters, use the following command:

```
python main.py dir_name
```

Mandatory:

`dir_name`: The directory name in which information about the run is saved by SMAC. This is a mandatory argument. 

Optional:

`--n_pre_runs`: Defines the number of runs for identifying the best architecture. 

`--n_post_runs`: Defines the number of runs to ascertain the best regularization or data augmentation for the model returned from the pre_run. 

`--pre_max_budget` and `--pre_min_budget`: Maximum and minimum number of epochs for the pre_run used by BOHB. Defaults are 3 and 1, respectively.

`--post_max_budget` and `--post_min_budget`: Maximum and minimum image sizes for the post_run used by BOHB. Defaults are 32 and 8, respectively.

`--n_workers`: Determines the number of workers to use for running SMAC. 

`--seed`: Seed value to ensure reproducibility. 

`--walltime_limit`: The maximum allowed time for the total run, given in seconds. Default is 6 hours.

## Evaluation

If you want to also evaluate the final model on the test after training using the `main.py` function, use:

```
python cfg_test.py dir_name
```

Mandatory:

`dir_name`: The directory name that was specified during the main.py run. This is a mandatory argument. 

Optional:

`--n_evals`: Determines the number of evaluations to run for the best configuration. Default is set to 3.

`--seed`: Seed value to ensure reproducibility. Ensure this matches the seed used during training for consistency.