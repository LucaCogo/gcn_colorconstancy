import os
import sys
import cv2
import numpy as np
import re
from skimage.util import view_as_windows
import tqdm
import ipdb

root = "../../Shi-Gehler"

def prepare_folder(root):
    # Copy files
    os.system(f"cp {os.path.join(root, 'all.txt')} ./all.txt")
    os.system(f"cp {os.path.join(root, 'train.txt')} ./train.txt")
    os.system(f"cp {os.path.join(root, 'val.txt')} ./val.txt")
    os.system(f"cp {os.path.join(root, 'test.txt')} ./test.txt")
    os.system(f"cp -r {os.path.join(root, 'metadata')} ./metadata")

    # Gen folders
    for f in ["1D", "5D - part 1", "5D - part 2", "5D - part 3", "5D - part 4"]:
        os.makedirs(f, exist_ok=True)

def read_img(img_path, mask_path):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
    img = img*mask
    img = cv2.resize(img, (224,224)) # cv2.resize(img, (2041, 1359))
    img = img / 255.0
    
    img = img.astype(np.float32).clip(0, 1).transpose(2,0,1)

    return img

def compute_histogram(x, patch_size):
    x = x.reshape(3, patch_size**2) 
    x = x*255
    h0, _ = np.histogram(x[0], bins=10, range=(0,255), density=True)
    h1, _ = np.histogram(x[1], bins=10, range=(0,255), density=True)
    h2, _ = np.histogram(x[2], bins=10, range=(0,255), density=True)
    h = np.concatenate([h0,h1,h2])

    return h

def compute_embeddings(patches_per_dim, patch_size):
    yy = np.tile(np.arange(0,patches_per_dim), (patches_per_dim,1))
    indices = np.array([yy.T, yy])

    positions = (indices * patch_size) + patch_size//2

    positions = positions.reshape(2,patches_per_dim**2).T

    return positions/(patches_per_dim*patch_size)

def extract_nodes(x, patch_size):
    c,h,w = x.shape
    
    num_patches = (h//patch_size)**2

    im_patches = view_as_windows(x, (3,patch_size,patch_size), step=patch_size).reshape((num_patches,3,patch_size,patch_size))

    # histograms = np.apply_along_axis(lambda a: compute_histogram(a, patch_size), 
    #                                  axis=1, 
    #                                  arr=im_patches.reshape((num_patches,3*patch_size**2))
    #                                 )
    mean_values = im_patches.mean(axis=(2,3))
    std_values = im_patches.std(axis=(2,3))
    perc_10 = np.percentile(im_patches, 10,axis=(2,3))
    perc_20 = np.percentile(im_patches, 20,axis=(2,3))
    perc_30 = np.percentile(im_patches, 30,axis=(2,3))
    perc_40 = np.percentile(im_patches, 40,axis=(2,3))
    perc_50 = np.percentile(im_patches, 50,axis=(2,3))
    perc_60 = np.percentile(im_patches, 60,axis=(2,3))
    perc_70 = np.percentile(im_patches, 70,axis=(2,3))
    perc_80 = np.percentile(im_patches, 80,axis=(2,3))
    perc_90 = np.percentile(im_patches, 90,axis=(2,3))

    patch_embeddings = compute_embeddings((h//patch_size), patch_size)
    
    patch_sizes = np.array([(patch_size**2) / (h*w)]*num_patches)

    nodes = np.concatenate([patch_embeddings, 
                            patch_sizes[:,None], 
                            mean_values, 
                            std_values, 
                            perc_10, 
                            perc_20, 
                            perc_30, 
                            perc_40, 
                            perc_50, 
                            perc_60, 
                            perc_70, 
                            perc_80, 
                            perc_90], 
                            axis=1)

    return nodes

def get_neighbors(matrix, i, j, distance=1):
    # Convert the input matrix to a numpy array
    matrix = np.array(matrix)
    
    # Get the number of rows and columns
    rows, cols = matrix.shape
    
    # Define the range of indices to look for neighbors based on the distance
    row_start, row_end = max(0, i - distance), min(rows, i + distance + 1)
    col_start, col_end = max(0, j - distance), min(cols, j + distance + 1)
    
    # Extract the submatrix of neighbors (including the element itself)
    sub_matrix = matrix[row_start:row_end, col_start:col_end]
    
    # Remove the center element (the original (i, j)) and return the neighbors
    # The flat index of the center element in the submatrix
    center_flat_index = (i - row_start) * sub_matrix.shape[1] + (j - col_start)
    
    # Use np.delete to remove the center element and return the remaining neighbors
    neighbors = np.delete(sub_matrix, center_flat_index)
    
    return neighbors


def generate_connectivity(n_nodes, distance):


    matrix = np.arange(n_nodes).reshape(int(n_nodes**(1/2)), int(n_nodes**(1/2)))
    connectivity = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            neighbors = get_neighbors(matrix, i, j, distance)
            for neighbor in neighbors:
                connectivity.append([matrix[i,j], neighbor])
    connectivity = np.array(connectivity).T

    return connectivity

def find_and_replace(file, find, replace):
    with open(file, "r") as r:
        text = r.read().replace(find, replace)
    with open(file, "w") as w:
        w.write(text)

if __name__ == "__main__":

    prepare_folder(root)

    # Read files_paths
    with open("all.txt", "r") as f:
            files_list = f.readlines()
    files_list = [f.strip() for f in files_list]

    patch_size = 16

    pbar = tqdm.tqdm(total=len(files_list))
    for f in files_list:
        img_path = os.path.join(root, f)
        mask_path = os.path.join(root, re.sub(r"/[0-9]*_","/masks/mask1_", f))
        dst_path = f

        img = read_img(img_path=img_path,
                    mask_path=mask_path)
        img = img
        

        # Generate graph data
        nodes = extract_nodes(img, patch_size=patch_size)
        single_connectivity = generate_connectivity(nodes.shape[0], 1)
        double_connectivity = generate_connectivity(nodes.shape[0], 2)
        full_connectivity = generate_connectivity(nodes.shape[0],4)
        
        # Save graph data
        np.savez(f.replace(".tiff",".npz"), 
                 nodes=nodes, 
                 single_connectivity=single_connectivity, 
                 double_connectivity=double_connectivity, 
                 full_connectivity=full_connectivity)


        pbar.update(1)


    # Change names in files 
    find_and_replace("all.txt", ".tiff", ".npz")
    find_and_replace("train.txt", ".tiff", ".npz")
    find_and_replace("val.txt", ".tiff", ".npz")
    find_and_replace("test.txt", ".tiff", ".npz")


        



