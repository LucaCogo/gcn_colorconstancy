import torch
import os
import torch.utils.data as data
import re
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, './auxiliary')
from auxiliary.augmentations import random_size_crop, random_crop, random_flip, random_rotation, random_jitter
import torch_geometric.data as gdata
import torch_geometric.loader as gloader


import ipdb

class ShiGehlerDataset(data.Dataset):

    def __init__(self, root_dir, files_list_path, rand_size_crop=None, rand_crop=None, rand_flip=False, rand_rotation=None, rand_jitter=None):
        
        super(ShiGehlerDataset, self).__init__()
        
        self.root_dir = root_dir
        self.files_list_path = files_list_path
        with open(files_list_path, "r") as f:
            self.files_list = f.readlines()
        self.files_list = [f.strip() for f in self.files_list]

        self.gt = np.array(pd.read_csv(os.path.join(root_dir, "metadata", "ColorCheckerData_REC_groundtruth.csv"), header=None))

        if rand_size_crop is not None:
            self.rand_size_crop = rand_size_crop
        if rand_crop is not None:
            self.rand_crop = rand_crop
        if rand_rotation is not None:
            self.rand_rotation = rand_rotation
        if rand_flip:
            self.rand_flip = rand_flip
        if rand_jitter is not None:
            self.rand_jitter = rand_jitter

    def __len__(self):
        return len(self.files_list)

    def read_img(self, img_path, mask_path):
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
        img = img*mask
        img = cv2.resize(img, (224,224)) # cv2.resize(img, (2041, 1359))
        img = img / 255.0
        
        img = img.astype(np.float32).clip(0, 1).transpose(2,0,1)

        return img

    def read_gt(self, gt_path):
        with open(gt_path, "r") as f:
            values = f.readlines()
        
        gt_ill = [eval(v) for v in values[0].strip().split(" ")] # BGR ill

        return np.array(gt_ill, dtype=np.float32) # RGB ill
    
    def read_rec_gt(self, number):
        return self.gt[number-1]
        
    
    def augmentations(self, img, ill):
        if hasattr(self, "rand_size_crop"):
            img = random_size_crop(img, self.rand_size_crop)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), (512,512))[0]
        if hasattr(self, "rand_crop"):
            img = random_crop(img, self.rand_crop)
        if hasattr(self, "rand_rotation"):
            img = random_rotation(img, self.rand_rotation)
        if hasattr(self, "rand_flip"):
            img = random_flip(img)
        if hasattr(self, "rand_jitter"):
            img, ill = random_jitter(img, ill, self.rand_jitter)

        return img, ill

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files_list[index])
        mask_path = os.path.join(self.root_dir, re.sub(r"/[0-9]*_","/masks/mask1_", self.files_list[index]))

        img = torch.Tensor(self.read_img(img_path, mask_path))
        gt_ill = torch.Tensor(self.read_rec_gt(int(img_path.split("/")[-1].split("_")[0]))) # Loading Recommended data annotation
        
        # # Uncomment if you want to use the old dataset annotation 
        # gt_path = os.path.join(self.root_dir, re.sub(r"[0-9]D.*/","ill_gt/", self.files_list[index].replace("tiff","txt")))
        # gt_ill = torch.Tensor(self.read_gt(gt_path))
        
        img, gt_ill = self.augmentations(img, gt_ill)

        item = {
            "img_path": img_path,
            "img": img,
            "gt_ill": gt_ill
        }

        return item

class ShiGehlerGraph(gdata.Dataset):

    def __init__(self, root_dir, files_list_path, connectivity):
        
        super(ShiGehlerGraph, self).__init__()
        
        self.connectivity = connectivity
        self.root_dir = root_dir
        self.files_list_path = files_list_path
        with open(files_list_path, "r") as f:
            self.files_list = f.readlines()
        self.files_list = [f.strip() for f in self.files_list]

        self.gt = np.array(pd.read_csv(os.path.join(root_dir, "metadata", "ColorCheckerData_REC_groundtruth.csv"), header=None))

    def read_img(self, img_path, mask_path):
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
        img = img*mask
        img = cv2.resize(img, (224,224)) # cv2.resize(img, (2041, 1359))
        img = img / 255.0
        
        img = img.astype(np.float32).clip(0, 1).transpose(2,0,1)

        return img

    def read_rec_gt(self, number):
        return self.gt[number-1]

    def len(self):
        return len(self.files_list)

    def get(self, index):
        graph_path = os.path.join(self.root_dir, self.files_list[index])

        gt_ill = torch.Tensor(self.read_rec_gt(int(graph_path.split("/")[-1].split("_")[0])))
        graph = np.load(graph_path)
        x = torch.Tensor(graph["nodes"])
        edge_index = torch.tensor(graph[self.connectivity], dtype=torch.int64)


        item = gdata.Data(x = x,
                          edge_index = edge_index,
                          y = gt_ill,
                          path = os.path.join(self.root_dir, self.files_list[index]))
        
        return item
