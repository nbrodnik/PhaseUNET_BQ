
import os
import pathlib

from skimage import io, exposure, morphology
import numpy as np
import support_functions as supp
import matplotlib.pyplot as plt
import performance_metrics as pm
import re
from tqdm import tqdm

def get_paths(d:str, ext:str=".tif", x_flag:str="x", y_flag:str="y") -> tuple:
    """Grabs all the images in a folder with a specific extension.
    Separates them into y_data and x_data based on flag.
    Sorted the data based on the number found at the end of the file name.
    NOTE: Make sure that the outputs are in the correct order ( [x0, x1,...] and [y0, y1, ...] not [y1, y0, ...]).
    They will be correct if the number is the last thing in the filename.
    
    Args:
             d (str): path to folder containing the images
           ext (str): extension to search for
        x_flag (str): identifier in the image file names that distinguishes the x images (the image data)
        y_flag (str): identifier in the image file names that distinguishes the y images (the mask)
    
    Returns:
        x_paths (list): all x-data paths
        y_paths (list): all y-data paths"""
    all_paths = [f for f in os.listdir(os.path.abspath(d)) if f.endswith(ext)]
    x_paths = [os.path.join(d, f) for f in all_paths if x_flag in f]
    y_paths = [os.path.join(d, f) for f in all_paths if y_flag in f]
    x_paths = sorted(x_paths, key=lambda x: int("".join(re.findall(r'\d+', x.split("/")[-1].split("_")[-1]))))
    y_paths = sorted(y_paths, key=lambda x: int("".join(re.findall(r'\d+', x.split("/")[-1].split("_")[-1]))))
    return x_paths, y_paths


def prep_input_data(x_paths, y_paths, train_size=0.8, seed=None, tilesize=256, clahe=True, clahe_clip_limit=0.03):
    """
    Prepare images for a UNET analysis.
    Args:
               x_paths (list): list of x_paths to read in
               y_paths (list): list of y_paths to read in
           train_size (flaot): fraction of dataset to be divided into training
                               (0.8 default, gives 80% training, 20% testing)
                   seed (int): random seed
                               (None default, aka no randomizing)
               tilesize (int): size of the (square) tiles to be created from the full images
                               (256 default)
                 clahe (bool): whether or not to perform CLAHE on the images
     clahe_clip_limit (float): the clip limit passed tothe CLAHE function if performed
                               (0.03 default)
    
    Returns
        y_train_stack (array): Array of training masks, y train data
        x_train_stack (array): Array of training data, x train data
         y_test_stack (array): Array of testing masks, y test data
         x_test_stack (array): Array of testing data, x test data
           tile_shape (array): The tiling shape used when creating image tiles
                               ex. 1100x1600 px images, 256 px tile size
                                   tiles are 256x256 px
                                   4 rows * 256 px tile height = 1024 px
                                   6 columns * 256 tile width = 1536 px
                                   usable area 1024x1536 px -> will crop data down to this size
                                   tile_shape = (4, 6)
    """
    # Check inputs
    if len(x_paths) != len(y_paths):
        raise ValueError("The number of x and y paths do not match: x({}), y({})".format(len(x_paths), len(y_paths)))
    # Get training data
    print("Preparing inputs...")
    print("\tNumber of paths:", len(x_paths))
    print("\tTraining size:", train_size)
    print("\tTile size size:", tilesize)
    y_stack = []
    x_stack = []
    for i in range(len(x_paths)):
        x = io.imread(x_paths[i])
        if clahe:
            x = exposure.equalize_adapthist(x, clip_limit=clahe_clip_limit)
        y = io.imread(y_paths[i]) > 0
        ##Setup model
        tile_shape = np.array(x.shape) // tilesize
        y_aug = supp.tile_and_augment(y, num_translations=0, rotations=8, exposure_adjust=0.15)
        x_aug = supp.tile_and_augment(x, num_translations=0, rotations=8, exposure_adjust=0.15)
        y_stack.append(y_aug)
        x_stack.append(x_aug)

    # Stack and correct dimensions and dtype for keras
    y_stack = np.vstack(y_stack)
    x_stack = np.vstack(x_stack)
    y_stack = np.expand_dims(y_stack, axis=-1).astype(np.float32)
    x_stack = np.expand_dims(x_stack, axis=-1).astype(np.float32)
    print("\tTotal number of tiles (including augmented):", x_stack.shape[0])

    # Randomize the training/testing data
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.choice(np.arange(y_stack.shape[0]), size=y_stack.shape[0], replace=False)
    else:
        indices = np.arange(y_stack.shape[0])
    # Apply randomization
    x_stack_r = x_stack[indices]
    y_stack_r = y_stack[indices]
    # Split into training and testing
    if train_size == 1:
        y_train_stack = y_stack_r
        x_train_stack = x_stack_r
        y_test_stack = np.empty(0)
        x_test_stack = np.empty(0)
    elif train_size == 0:
        y_train_stack = np.empty(0)
        x_train_stack = np.empty(0)
        y_test_stack = y_stack_r
        x_test_stack = x_stack_r
    else:
        y_train_stack = y_stack_r[:int(y_stack.shape[0] * train_size)]
        x_train_stack = x_stack_r[:int(x_stack.shape[0] * train_size)]
        y_test_stack = y_stack_r[int(y_stack.shape[0] * train_size):]
        x_test_stack = x_stack_r[int(x_stack.shape[0] * train_size):]
    return (y_train_stack, x_train_stack, y_test_stack, x_test_stack, tile_shape)


def test(model, y_test, x_test, tile_shape, name="model", batch_size=1, tilesize=256):
    # Run on test data
    y_p = model.predict(x_test, verbose=0, batch_size=batch_size)
    # Get best threshold (coarse)
    print("Tuning theta...")
    thetas = np.arange(0.05, 1.0, 0.05)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas[:, None, None, None]).astype(np.float32)
    ious = np.array([pm.Jaccard_coef(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas)), desc="IoU (coarse)")])
    best_theta = thetas[np.argmax(ious)]
    # Get best threshold (fine)
    thetas2 = np.arange(max(0.001, best_theta - 0.05), min(1.0, best_theta + 0.05), 0.01)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas2),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas2[:, None, None, None]).astype(np.float32)
    ious2 = np.array([pm.Jaccard_coef(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas2)), desc="IoU (fine)")])
    best_theta = thetas2[np.argmax(ious2)]
    # Get best threshold (finer)
    thetas3 = np.arange(max(0.0001, best_theta - 0.005), min(1.0, best_theta + 0.005), 0.001)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas3),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas3[:, None, None, None]).astype(np.float32)
    ious3 = np.array([pm.Jaccard_coef(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas3)), desc="IoU (finer)")])
    best_theta = thetas3[np.argmax(ious3)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(thetas, ious, color="blue", marker="o", lw=2, markersize=10)
    ax.plot(thetas2, ious2, color="red", marker="x", lw=2, markersize=10)
    ax.plot(thetas3, ious3, color="green", marker="+", lw=2, markersize=10)
    ax.axvline(best_theta, color="k", linestyle="--")
    ax.set_xlabel("Threshold", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20, color="blue")
    plt.title(f"{name} IoU vs Threshold")
    plt.tight_layout()
    plt.savefig("./theta_tuning.png", dpi=300)
    plt.close("all")

    print("   -> " + name + " Best theta:", best_theta)
    y_pred = np.where(y_p > best_theta, 1, 0).astype(np.float32)
    y_pred_1d = y_pred.reshape(-1)
    y_test_1d = y_test.reshape(-1)
    y_pred_combined = supp.combine_tiles(y_pred[:, :, :, 0], tile_shape)
    y_test_combined = supp.combine_tiles(y_test[:, :, :, 0], tile_shape)
    x_test_combined = supp.combine_tiles(x_test[:, :, :, 0], tile_shape)

    supp.compare_n_ims([y_test_combined, y_pred_combined, x_test_combined], ['Ground truth', 'Predicted', "Data"], show=False, grid=tilesize)
    plt.savefig(f'./{name}_Test-Results.png', dpi=300)
    recall = pm.Recall(y_test_1d, y_pred_1d)
    precision = pm.Precision(y_test_1d, y_pred_1d)
    f1 = pm.F1(y_test_1d, y_pred_1d)
    iou = pm.Jaccard_coef(y_test, y_pred)
    return (recall, precision, f1, iou, best_theta)
