
This is a fork of [gluonts-hierarchical-ICML-2021](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021) 

## Setup
Download this code.
```
pip install --upgrade mxnet
cd gluonts-hierarchical-ICML-2021
pip install -e .
```

## Running

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method DeepVAR
```
where dataset is one of `{labour, traffic, tourism, wiki}` .


This will run the selected method 5 times on the selected dataset with the hyperparameters used in the paper.