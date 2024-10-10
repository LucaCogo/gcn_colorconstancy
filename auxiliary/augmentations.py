import random
import numpy as np
import torch
from torchvision.transforms import RandomRotation

import ipdb

def random_size_crop(img, crop_range):
    c, h, w = img.shape

    scale = random.random() * (crop_range[1]-crop_range[0]) + crop_range[0]
    crop = [int(h*scale), int(w*scale)]

    y  = random.randint(0, h-crop[0])
    x = random.randint(0, w-crop[1])

    cropped = img[:, y:y+crop[0], x:x+crop[1]]
    return cropped

def random_crop(img, size):
    _, h, w = img.shape
    y  = random.randint(0, h-size[0])
    x = random.randint(0, w-size[1])

    return img[:, y:y+size[0], x:x+size[1]]

def random_flip(img):
    if np.random.rand() > 0.5:
        img = torch.flip(img, [2])
    if np.random.rand() > 0.5:
        img = torch.flip(img, [1])

    return img

def random_rotation_90(img):
    if np.random.rand() > 0.5:
        img = torch.rot90(img, 1, [1, 2])
    if np.random.rand() > 0.5:
        img = torch.rot90(img, 2, [1, 2])
    if np.random.rand() > 0.5:
        img = torch.rot90(img, 3, [1, 2])

    return img

def random_rotation(img, angle):
    rotater = RandomRotation(degrees=angle)
    rotated_img = rotater(img)
    return rotated_img

def random_jitter(img, ill, _range):

    scaling_rgb = torch.Tensor([random.random()*(_range[1]-_range[0]) + _range[0] for _ in range(3)])

    img = torch.clamp(img * scaling_rgb[:,None,None], 0, 1)
    ill = torch.clamp(ill * scaling_rgb, 0, 1)

    return img, ill