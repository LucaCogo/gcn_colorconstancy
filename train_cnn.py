import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from datasets import ShiGehlerDataset
from auxiliary.losses import AngularError
from auxiliary.utils import correct_img, save_viz_3, rad2deg, format_msg, plot_stats
from auxiliary.utils import seed_everything
from auxiliary.options import get_options, make_experiment
import tqdm
import argparse

import ipdb


def train(opt):

    print(f"Using device: {opt.device}")
    seed_everything(opt.seed)
    os.makedirs(opt.exp_dir, exist_ok=True)

    train_dataset, train_dataloader, val_dataset, val_dataloader, model = make_experiment(opt, mode="train")

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=opt.lr)

    criterion = AngularError()

    best_loss = np.inf
    train_losses = []
    val_losses = []
    early_stop = 0
    for epoch in range(opt.n_epochs):
        model.train()
        train_epoch_losses = []
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=train_dataset.__len__(), desc=f"Train Epoch {epoch+1}/{opt.n_epochs}")
        for i, batch in enumerate(train_dataloader):
            # Forward pass
            img = batch["img"].to(opt.device)
            gt_ill = batch["gt_ill"].to(opt.device)
            
            optimizer.zero_grad() # Reset the gradients
            pred_ill = model(img)
            
            loss, _ = criterion(pred_ill, gt_ill) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            train_epoch_losses.append(loss.item())

            pbar.update(batch["img"].shape[0])

        pbar.close()
        train_losses.append(np.mean(train_epoch_losses)) # Average the losses

        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(val_dataloader), total=val_dataset.__len__(), desc=f"Val Epoch {epoch+1}/{opt.n_epochs}")
            for i, batch in enumerate(val_dataloader):
                # Forward pass 
                img = batch["img"].to(opt.device)
                gt_ill = batch["gt_ill"].to(opt.device)
                pred_ill = model(img)
                loss, _ = criterion(pred_ill, gt_ill) # Compute loss
                
                val_epoch_losses.append(loss.item())

                if i == 0:
                    img = img.detach().cpu()
                    gt_ill = gt_ill.detach().cpu()
                    pred_ill = pred_ill.detach().cpu()

                    pred_correction = correct_img(img[0], pred_ill[0])
                    gt_correction = correct_img(img[0], gt_ill[0])
                    save_viz_3( img[0]**(1/2.2),
                                pred_correction**(1/2.2),
                                gt_correction**(1/2.2),
                                epoch+1,
                                opt.exp_dir,
                                f"epoch_{epoch+1}({np.round(180*loss.item()/np.pi,2)}°).png") # Save the visualizations if required
                
                pbar.update(batch["img"].shape[0])
            
            pbar.close()
            val_mean = np.mean(val_epoch_losses)
            val_best = np.min(val_epoch_losses)
            val_worst = np.max(val_epoch_losses)
            val_median = np.median(val_epoch_losses)
            val_best25p = np.percentile(val_epoch_losses, 25)
            val_worst25p = np.percentile(val_epoch_losses, 75)
            val_trimean = (val_best25p + 2*val_median + val_worst25p)/4
            val_losses.append(val_mean)

        # Print results of the epoch
        epoch_end_message = [
            f"Epoch {epoch+1}/{opt.n_epochs} finished:",
            f"| Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_mean:.4f} |",
            f"| Train AE: {rad2deg(train_losses[-1]):.4f} | Val AE: {rad2deg(val_mean):.4f} |",
            f"| Mean {rad2deg(val_mean):.4f} | Med. {rad2deg(val_median):.4f} | Trimean {rad2deg(val_trimean):.4f} || B-25 {rad2deg(val_best25p):.4f} | W-25 {rad2deg(val_worst25p):.4f} || Best {rad2deg(val_best):.4f} | Worst {rad2deg(val_worst):.4f} |"
        ]
        print(format_msg(epoch_end_message))
        with open(os.path.join(opt.exp_dir, "train_log.txt"), "a") as f:
            f.write(format_msg(epoch_end_message)+"\n")

        plot_stats(train_losses, val_losses, opt.exp_dir)

        torch.save(model.state_dict(), os.path.join(opt.exp_dir,"last.pth"))
        if val_losses[-1] < best_loss:
            early_stop = 0
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), os.path.join(opt.exp_dir,"best.pth"))
        else:
            early_stop += 1
        
        if opt.early_stop > 0 and early_stop >= opt.early_stop:
            break



def test(opt):
    
    print(f"Using device: {opt.device}")
    
    os.makedirs(opt.exp_dir, exist_ok=True)

    
    test_dataset, test_dataloader, model = make_experiment(opt, "test")
    model.load_state_dict(torch.load(os.path.join(opt.exp_dir, "best.pth")))

    criterion = AngularError()
  
    model.eval()
    test_losses = []
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(test_dataloader), total=test_dataset.__len__(), desc=f"Testing")
        for i, batch in enumerate(test_dataloader):
            # Forward pass 
            img = batch["img"].to(opt.device)
            gt_ill = batch["gt_ill"].to(opt.device)
            pred_ill = model(img)
            loss, _ = criterion(pred_ill, gt_ill) # Compute loss
            
            test_losses.append(loss.item())

            img = img.detach().cpu()
            gt_ill = gt_ill.detach().cpu()
            pred_ill = pred_ill.detach().cpu()

            pred_correction = correct_img(img[0], pred_ill[0])
            gt_correction = correct_img(img[0], gt_ill[0])
            save_viz_3( img[0]**(1/2.2),
                        pred_correction**(1/2.2),
                        gt_correction**(1/2.2),
                        epoch = 0,
                        experiment_dir = opt.exp_dir,
                        name=f"{batch['img_path'][0].split('/')[-1].replace('/','_')}({np.round(180*loss.item()/np.pi,2)}°).png", # Save the visualizations if required
                        viz_dir = "test_viz")
                
            pbar.update(batch["img"].shape[0])
        
        pbar.close()
        test_mean = np.mean(test_losses)
        test_best = np.min(test_losses)
        test_worst = np.max(test_losses)
        test_median = np.median(test_losses)
        test_best25p = np.percentile(test_losses, 25)
        test_worst25p = np.percentile(test_losses, 75)
        test_trimean = (test_best25p + 2*test_median + test_worst25p)/4
        test_losses.append(test_mean)

    # Print results of the epoch
    epoch_end_message = [
        f"Testing finished:",
        f"| Test Loss: {test_mean:.4f} |",
        f"| Test AE: {rad2deg(test_mean):.4f} |",
        f"| Mean {rad2deg(test_mean):.4f} | Med. {rad2deg(test_median):.4f} | Trimean {rad2deg(test_trimean):.4f} || B-25 {rad2deg(test_best25p):.4f} | W-25 {rad2deg(test_worst25p):.4f} || Best {rad2deg(test_best):.4f} | Worst {rad2deg(test_worst):.4f} |"
    ]
    print(format_msg(epoch_end_message))
    with open(os.path.join(opt.exp_dir, "test_log.txt"), "w") as f:
        f.write(format_msg(epoch_end_message)+"\n")



if __name__ == "__main__":
    opt = get_options() # Use argument parser to get training options
    train(opt)
    test(opt)
    