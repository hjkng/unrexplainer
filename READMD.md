# UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models

This repository is the official implementation of the UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models

## Overview

![overview](https://anonymous.4open.science/r/unr0929/overview.jpg)


## Requirements
    • python == 3.9.7
    • pytorch == 1.13.1
    • pytorch-cluster == 1.6.0
    • pyg == 2.2.0
    • pytorch-scatter == 2.1.0
    • pytorch-sparse == 0.6.16
    • cuda == 11.7.1
    • numpy == 1.23.5
    • tensorboardx == 2.2
    • networkx == 3.0
    • scikit-learn == 1.1.3
    • scipy == 1.9.3
    • pandas == 1.5.2

## Training original models

To train the original GNN models for the datasets in the paper, run the following command:

```
python train_graphsage.py --dataset=='Cora' --model=='GraphSage'
```

## Explanation

To explain the trained model, run the following command:

```
python explain_model.py --dataset=='Cora' --model=='GraphSage'
```
