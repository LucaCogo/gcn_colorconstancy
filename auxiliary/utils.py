import numpy as np
import torch
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import ipdb

def correct_img(img, illum):
    """
    Applies Von Kries illuminant correction to a batch of images using corresponding illuminants
    """
    corrected = torch.clamp(img / (illum[:,None,None]*np.sqrt(3) + 1e-10), 0, 1)
    normalized = corrected / corrected.max()
    
    return normalized

def save_viz_3(x,y,z, epoch, experiment_dir, name = None, viz_dir="visualizations"):
    """
    Utility function to save visualizations in the experiment results folder
    """

    # Create the visualization directory
    viz_dir = os.path.join(experiment_dir, viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    name = f"epoch_{epoch}.png" if name is None else name

    x = x.detach().cpu().permute(1,2,0).numpy()
    y = y.detach().cpu().permute(1,2,0).numpy()
    z = z.detach().cpu().permute(1,2,0).numpy()
    img = np.concatenate([x,y,z], axis=1)
    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(viz_dir, name), img)

def rad2deg(x):
    """
    Converts an angle from radians to degrees 
    """

    return 180*x/np.pi

def format_msg(messages):
    """
    Formats the text written on epoch end for experiments logging
    """

    out = ""
    for message in messages:
        out += ("-"*len(message)) + "\n"
        out += message + "\n"

    out+= ("-"*len(messages[-1])) + "\n"

    return out

def plot_stats(train_losses, val_losses, experiment_dir):
    """
    Create a plot of the loss and metrics and saves it to a multi-page pdf file
    """

    exp_stats = [{"epoch" : e+1,
                  "train_loss": v[0], 
                  "val_loss" : v[1]} for e,v in enumerate(zip(train_losses, val_losses))]


    df = pd.DataFrame(exp_stats)
    pdf_path = os.path.join(experiment_dir, "plots.pdf")


    with PdfPages(pdf_path) as pdf:
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="Train")
        plt.plot(df["epoch"], df["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")


        plt.figure()
        plt.plot(df["epoch"], rad2deg(df["train_loss"]), label="Train")
        plt.plot(df["epoch"], rad2deg(df["val_loss"]), label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Angular Error")
        plt.legend()
        plt.title("Angular Error")

        for i in plt.get_fignums():
            pdf.savefig(plt.figure(i))
        plt.close("all")

def seed_everything(seed):
    """
    Makes experiments deterministic
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.utils.deterministic.fill_uninitialized_memory_(True)