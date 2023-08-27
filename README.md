[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/T_Fzxg5j)
# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project

This repository is one approach to solve the assignment of the AutoML lecture at the university of Freiburg in Summer 2023.
The objective was to find a good CNN architecture for solving an image classification tasks with tools from AutoML on a given budget (6 hours).
The used approach was using different fidelities at different stages at the optimization process using SMAC (https://github.com/automl/SMAC3)

### Training

To run the full training pipeline, use the following command:

```
python main.py directory
```

Where directory is the output folder of the SMAC run

If you want to also test the model on the test test after training you may use 

```
python cfg_test.py directory
```

Where directory is the output folder of the SMAC run (same argument of the main.py script)