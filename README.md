# Parametric Bayesian Learning Games
Official implementation of our ICML'22 paper ["On the Convergence of the Shapley Value in Parametric Bayesian Learning Games"](https://arxiv.org/abs/2205.07428) (__21.9%__ acceptance rate).

## Requirements
1. Linux machine (experiments were run on Ubuntu 18.04.5 LTS and Ubuntu 20.04.2 LTS)
2. Anaconda (alternatively, you may install the packages in `environment.yml` manually)

## Setup
1. Run the following command to install the required Python packages into a new environment named PBLG using Anaconda.
```shell
conda env create -f environment.yml
```

## Running experiments
In the main directory,
1. Change current environment to the PBLG environment.
```shell
conda activate PBLG
```
2. Run the desired experiment. Files with names `exp_*.py` are scripts for different experiments. 
- For instance, `exp_CaliH-P1.py` runs the experiment on the California housing dataset by varying the hyperparameters w.r.t. Player 1 (P1).
- Similarly, `exp_CaliH-multi.py` runs the experiment on the California housing dataset with multiple players (i.e., 4).

## License
This code is released under the MIT License.

## Citing our paper
If you find our paper relevant or use our code in your research, please consider citing our paper:
```
@InProceedings{Agussurja2022,
  title={Incentivizing collaboration in machine learning via synthetic data rewards},
  author={Lucas Agussurja and Xinyi Xu and Bryan Kian Hsiang Low},
  booktitle={Proc. ICML},
  year={2022}
}
```

