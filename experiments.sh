#!/bin/bash

python train_cnn.py --exp_name AlexNet --model AlexNet --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results;

python train_graph.py --exp_name GCN_single --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity single_connectivity;

python train_graph.py --exp_name GCN_double --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity double_connectivity;

python train_graph.py --exp_name GCN_full --model GCN --seed 42 --device 0 --n_epochs 500 --early_stop 10 --n_workers 8 --lr 0.00001 --batch_size 2 --save_dir results --connectivity full_connectivity;
