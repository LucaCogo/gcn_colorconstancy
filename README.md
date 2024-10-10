# Computational Color Constancy using Graph Convolutional Networks

This repository contains the code to reproduce experiments conducted for the exam of the Ph.D. course "Geometry Processing and Machine Learning for Geometric Data", held at University of Milano-Bicocca by Simone Melzi and Riccardo Marin in November 2023.

The project investigates the adoption of graph convolutional networks for the color constancy task.
More details regarding the conducted experiments can be found in the slides.

## Repository organization:

```
project
│   README.md
│   slides.pdf: presentation explaining the project and the experiments results
│   requirements.txt
│   train_cnn.py: training and testing script for training the baseline CNN
│   train_gcn.py: training and testing script for training the GCN
│
└───data:
│   │
│   └───Shi-Gehler: color constancy dataset adopted for experiments
│   │
│   └───Shi-Gehler-graph
│       └───patches: Shi-Gehler dataset where images are converted into graphs
│
└───datasets: contains torch Dataset classes for data loading
│
└───models: contains torch models for CNN and GCN
│
└───auxiliary: contains utility functions for the experiments
│
└───results: contains experiments results
```

## Getting started
Before running the experiments, please make sure to have installed all the libraries reported in requirements.txt.
Otherwise, run `pip install -r requirements.txt` to install all the missing dependencies.

The `data` folder needs to be populated with the Shi-Gehler dataset with the following procedure:

1) Download the dataset from <a href="https://drive.google.com/file/d/1YNXsEuIqy64bq8_nmxpbmClsPLosnCAE/view?usp=sharing">here</a> and unzip it inside the data folder
2) Run the `generate.py` script inside `data/Shi-Gehler-graph/`to generate graph data

## Running the experiments
You can re-run all the experiments running the ```experiments.sh``` file.
Otherwise, here are some snippets for running single experiments:

```
# Train and test the baseline CNN 
python train_cnn.py --exp_name AlexNet --model AlexNet --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results;

# Train and test the GCN with single connectivity
python train_graph.py --exp_name GCN_single --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity single_connectivity;

# Train and test the GCN with double connectivity
python train_graph.py --exp_name GCN_double --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity double_connectivity;

# Train and test the GCN with full connectivity
python train_graph.py --exp_name GCN_full --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity full_connectivity;
```


