<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

```plot hessian activity condition number
PYTHONPATH=. python scripts/plot_activity_hessian.py
```
*EQProp preferred setting*
python src/train.py model/net=ep_mnist model.net.beta=1 model.net.solver.amp_factor=6.0 model/optimizer=adamw trainer=gpu 

python src/train.py -m model/net=ep_mnist hparams_search=mnist_optuna trainer=gpu model/optimizer=adamw

python src/train.py model/net=ep_mnist model.net.beta=4.2824 model.net.solver.amp_factor=9.5824 model/optimizer=adamw trainer=gpu model.optimizer.lr=0.001695 data.batch_si
ze=128

python src/train.py model/net=ep_mnist model.net.beta=1.717 model.net.solver.amp_factor=11.879 model.optimizer.lr=0.00125 model/optimizer=adamw trainer=gpu logger=[csv,wandb] data.batch_size=256

make plot-ill-conditioning
make plot-hessian-condition

python src/train.py model/net=ep_mnist model.net.beta=1.1221 model.net.solver.amp_factor=12.441 model.optimizer.lr=0.005738 model/optimizer=adamw trainer=gpu logger=[csv,wandb] data.batch_size=256 #128-64 0.5~