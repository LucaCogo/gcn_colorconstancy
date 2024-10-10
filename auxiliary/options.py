import argparse
import torch
from datetime import datetime
import os
from datasets.shigehler_dataset import ShiGehlerDataset, ShiGehlerGraph
from models.simple_cnn import AlexNet
from models.simple_gcn import GCN
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricLoader

def get_options():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--model", type=str, help="Choose model to train (FC4, CNN, ...)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",type=int, default=0, help="Choose device number (-1 for cpu)")
    parser.add_argument("--n_epochs", type=int, help="Choose number of epochs for training")
    parser.add_argument("--n_workers",type=int, default=8, help="Choose number of workers (use 0 for dataloader debugging)")
    parser.add_argument("--lr",type=float, help="Choose learning rate")
    parser.add_argument("--batch_size",type=int, help="Choose batch size during training (validation is set to 1)")
    parser.add_argument("--save_dir", default="results", help="Name of the directory where to save experiments data")
    parser.add_argument("--early_stop", type=int, default=-1, help="Define patience for early-stopping, if -1 no early-stopping is used")
    parser.add_argument("--connectivity", default=None, help="Connectivity scheme for graph experiments (\{single;double;full\}_connectivity)")


    opt = parser.parse_args()
    opt.exp_dir = os.path.join(opt.save_dir, datetime.now().strftime("%y%m%d_%H%M%S")+ "_" + opt.exp_name)
    opt.device = torch.device(f"cuda:{opt.device}" if opt.device >= 0 and opt.device >= torch.cuda.device_count() - 1 else "cpu")
    
    return opt

def make_experiment(opt, mode):
    if opt.model in ["AlexNet"]:
        return make_cnn_experiment(opt, mode)
    elif opt.model in ["GCN"]:
        return make_gcn_experiment(opt, mode)
    
def make_cnn_experiment(opt, mode):

    if mode.lower() == "train":
        train_dataset = ShiGehlerDataset(root_dir="data/Shi-Gehler/", files_list_path="data/Shi-Gehler/train.txt",
                                        rand_size_crop=None, rand_flip=None, rand_rotation=None, rand_jitter=None)
        val_dataset =  ShiGehlerDataset(root_dir="data/Shi-Gehler/", files_list_path="data/Shi-Gehler/val.txt")
        
        g = torch.Generator()
        g.manual_seed(opt.seed)
        train_dataloader = DataLoader(train_dataset,
                                    num_workers=opt.n_workers,
                                    batch_size=opt.batch_size,
                                    generator=g,
                                    shuffle=True
                                    )
        
        val_dataloader = DataLoader(val_dataset,
                                    num_workers=opt.n_workers,
                                    batch_size=1,
                                    generator=g,
                                    shuffle=False
                                    )
        model = eval(opt.model)().to(opt.device)

        return train_dataset, train_dataloader, val_dataset, val_dataloader, model
    else:
        test_dataset = ShiGehlerDataset(root_dir="data/Shi-Gehler/", files_list_path="data/Shi-Gehler/test.txt")
        test_dataloader = DataLoader(test_dataset,
                                    num_workers=opt.n_workers,
                                    batch_size=1,
                                    shuffle=False
                                    )
        model = eval(opt.model)().to(opt.device)
        
        return test_dataset, test_dataloader, model
    
def make_gcn_experiment(opt, mode):
    if mode.lower() == "train":
        train_dataset = ShiGehlerGraph(root_dir="data/Shi-Gehler-graph/patches", files_list_path="data/Shi-Gehler-graph/patches/train.txt", connectivity=opt.connectivity)
        val_dataset =  ShiGehlerGraph(root_dir="data/Shi-Gehler-graph/patches", files_list_path="data/Shi-Gehler-graph/patches/val.txt", connectivity=opt.connectivity)

        g = torch.Generator()
        g.manual_seed(opt.seed)
        train_dataloader = GeometricLoader(train_dataset,
                                           num_workers=opt.n_workers,
                                           batch_size=opt.batch_size,
                                           generator=g,
                                           shuffle=True
                                           )
        
        val_dataloader = GeometricLoader(val_dataset,
                                    num_workers=opt.n_workers,
                                    batch_size=1,
                                    generator=g,
                                    shuffle=False
                                    )
        
        model = eval(opt.model)().to(opt.device)
        
        return train_dataset, train_dataloader, val_dataset, val_dataloader, model
    else:
        test_dataset = ShiGehlerGraph(root_dir="data/Shi-Gehler-graph/patches", files_list_path="data/Shi-Gehler-graph/patches/test.txt", connectivity=opt.connectivity)
        test_dataloader = GeometricLoader(test_dataset,
                                    num_workers=opt.n_workers,
                                    batch_size=1,
                                    shuffle=False
                                    )
        model = eval(opt.model)().to(opt.device)
        
        return test_dataset, test_dataloader, model