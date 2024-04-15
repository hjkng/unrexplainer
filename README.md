# UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models

This repository is the official implementation of the UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models

## Overview

![overview](https://github.com/hjkng/unr/blob/main/explainer/overview.jpg)

## Setup
We provide an environment.yml file to create a Conda environment:

```
conda env create -f environment.yml
conda activate unr
```


## Explain

To explain the trained model, run the following command:

```
python explain_gnns.py --dataset syn1 --model graphsage --task node
python explain_gnns.py --dataset syn3 --model graphsage --task node
python explain_gnns.py --dataset syn4 --model graphsage --task node
python explain_gnns.py --dataset Cora --model graphsage --task link
python explain_gnns.py --dataset CiteSeer --model graphsage --task link
python explain_gnns.py --dataset PubMed --model dgi --task node
```


## Evaluate

```
python evaluate_expl_syn.py --dataset syn1 --model graphsage --task node

python evaluate_expl_syn.py --dataset syn3 --model graphsage --task node

python evaluate_expl_syn.py --dataset syn4 --model graphsage --task node

python evaluate_expl.py --dataset Cora --model graphsage --task link

python evaluate_expl.py --dataset CiteSeer --model graphsage --task link

python evaluate_expl.py --dataset PubMed --model dgi --task node
```


## Train gnns

To train the original GNN models for the datasets in the paper, run the following command:

```
python train_gnns.py --dataset syn1 --model graphsage --task node
python train_gnns.py --dataset syn3 --model graphsage --task node
python train_gnns.py --dataset syn4 --model graphsage --task node
python train_gnns.py --dataset Cora --model graphsage --task link
python train_gnns.py --dataset CiteSeer --model graphsage --task link
python train_gnns.py --dataset PubMed --model dgi --task node
```
