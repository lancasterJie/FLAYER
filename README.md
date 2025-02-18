# Introduction

Source code for "FLAYER: Optimizing Personalized Federated Learning through Adaptive Layer-Wise Learning".

# Environments

With the installed conda, we can run this platform in a conda virtual environment called fl_torch that contains the required dependencies.

```bash
conda env create -f env_linux.yaml  # for Linux
```

# Dataset

This code only provides the case where the CIFAR100 dataset is divided into 20 client data using the Dirichlet distribution. If you want to obtain more segmentation results of the dataset, please refer to the https://github.com/TsingZ0/PFLlib website.

# Run

Execution examples have been written in the run_me.sh file for different datasets and models. When executing, only the required command line needs to be retained, and other command lines are commented.

```bash
cd ./system
sh run_me.sh
```
