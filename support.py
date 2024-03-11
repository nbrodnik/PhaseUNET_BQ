import os
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from skimage import io, exposure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import MultipleLocator
import performance_metrics as pm
import re
from tqdm.auto import tqdm



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
    y_paths = sorted(y_paths, key=lambda y: int("".join(re.findall(r'\d+', y.split("/")[-1].split("_")[-1]))))
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
        x = (x - x.min()) / (x.max() - x.min())
        y = io.imread(y_paths[i]) > 0
        ##Setup model
        y_aug, _ = tile_and_augment(y, num_translations=2, rotations=8, exposure_adjust=0.15)
        x_aug, tile_shape = tile_and_augment(x, num_translations=2, rotations=8, exposure_adjust=0.15)
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


def tune_theta(y_p, y_test, name=""):
    """Tune the threshold for the model.
    Args:
        y_p: the predicted masks
        y_test: the test masks
        name: the name of the model
    Returns:
        best_theta: the best threshold for the model"""
    thetas = np.arange(0.05, 1.0, 0.05)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas[:, None, None, None]).astype(np.float32)
    ious = np.array([pm.IoU(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas)), desc="IoU (coarse)")])
    best_theta = thetas[np.argmax(ious)]
    # Get best threshold (fine)
    thetas2 = np.arange(max(0.001, best_theta - 0.05), min(1.0, best_theta + 0.05), 0.01)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas2),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas2[:, None, None, None]).astype(np.float32)
    ious2 = np.array([pm.IoU(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas2)), desc="IoU (fine)")])
    best_theta = thetas2[np.argmax(ious2)]
    # Get best threshold (finer)
    thetas3 = np.arange(max(0.0001, best_theta - 0.005), min(1.0, best_theta + 0.005), 0.001)
    y_p_broadcast = np.broadcast_to(y_p[:, :, :, 0], (len(thetas3),) + y_p.shape[:-1])
    y_p_all_threshold = (y_p_broadcast > thetas3[:, None, None, None]).astype(np.float32)
    ious3 = np.array([pm.IoU(y_test, y_p_all_threshold[i][:, :, :, None]) for i in tqdm(range(len(thetas3)), desc="IoU (finer)")])
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
    return best_theta


def test(model, y_test, x_test, tile_shape, name="model", batch_size=1, tilesize=256):
    """Test a model on the test data.
    Will do theta tuning and will save the results to a file.
    Args:
        model: the model to test
        y_test: the test masks
        x_test: the test data
        tile_shape: the shape of the tiles
        name: the name of the model
        batch_size: the batch size to use
        tilesize: the size of the tiles
    Returns:
        None"""
    # Run on test data
    y_p = model.predict(x_test, verbose=0, batch_size=batch_size)
    # Get best threshold (coarse)
    print("Tuning theta...")
    best_theta = tune_theta(y_p, y_test, name)

    print("   -> " + name + " Best theta:", best_theta)
    y_pred = np.where(y_p > best_theta, 1, 0).astype(np.float32)
    y_pred_1d = y_pred.reshape(-1)
    y_test_1d = y_test.reshape(-1)
    y_pred_combined = untile(y_pred[:np.prod(tile_shape), :, :, 0], tile_shape)
    y_test_combined = untile(y_test[:np.prod(tile_shape), :, :, 0], tile_shape)
    x_test_combined = untile(x_test[:np.prod(tile_shape), :, :, 0], tile_shape)

    compare_n_ims([y_test_combined, y_pred_combined, x_test_combined],
                  ['Ground truth', 'Predicted', "Data"],
                  show=False,
                  grid_line_spacing=None)
    plt.savefig(f'./{name}_Test-Results.png', dpi=300)
    recall = pm.Recall(y_test_1d, y_pred_1d)
    precision = pm.Precision(y_test_1d, y_pred_1d)
    f1 = pm.F1(y_test_1d, y_pred_1d)
    iou = pm.IoU(y_test, y_pred)

    print("Recall: {}".format(recall))
    print("Precision: {}".format(precision))
    print("F1: {}".format(f1))
    print("IoU: {}".format(iou))

    with open("./Results.txt", "a") as f:
        line = f"{name} -Theta:{best_theta:.3f}- Recall:{recall:.4f} Precision:{precision:.4f} F1:{f1:.4f} IoU:{iou:.4f}\n"
        f.write(line)


def compare_n_ims(im, titles=None, show=False, grid_line_spacing=None):
    """Function for comparing n images side by side.
    Args:
        im: list of images to compare
        titles: list of titles for the images
        show: boolean of whether to show the images
        grid: int of the grid size to use for the images
    Returns:
        None"""
    fig = plt.figure(4, figsize=(21,7))
    axes = []
    for i in range(len(im)):
        ax = fig.add_subplot(1, len(im), i+1)
        ax.imshow(im[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i])
        axes.append(ax)
        if grid_line_spacing is not None:
            ax.xaxis.set_major_locator(MultipleLocator(grid_line_spacing))
            ax.yaxis.set_major_locator(MultipleLocator(grid_line_spacing))
    plt.tight_layout()
    if show:
        plt.show()


def tileslice(img, tilesize, offset=(0,0)):
    """Function for tiling an image into 256x256 tiles.
    Args:
        imginput: np.array of shape (H,W)
        tiles: np.array of length 2, containing x direction number of tiles and y dir number of tiles
        offset: np.array of length 2, containing x and y offset of image from top left corner of image
    Returns:
        np.array of shape (tiles[0]*tiles[1],256,256)"""
    tile_shape = np.array(img.shape) // tilesize
    s = (slice(offset[0], offset[0] + tile_shape[0] * tilesize),
         slice(offset[1], offset[1] + tile_shape[1] * tilesize))
    out = img[s]
    out = np.array(np.split(out, tile_shape[1], axis=-1))
    out = np.array(np.split(out, tile_shape[0], axis=-2))
    out = np.reshape(out, (tile_shape[0] * tile_shape[1], tilesize, tilesize))
    return out, tile_shape


def untile(imgs, tile_shape):
    """Function for untiling an image from 256x256 tiles.
    Args:
        img: np.array of shape (tiles[0]*tiles[1],tilesize,tilesize)
        tile_shape: np.array of length 2, containing row direction number of tiles and column dir number of tiles
    Returns:
        np.array of shape (tiles[0]*tilesize,tiles[1]*tilesize)"""
    imgs = np.reshape(imgs, (tile_shape[0], tile_shape[1], imgs.shape[1], imgs.shape[2]))
    imgs = np.concatenate(imgs, axis=-2)
    imgs = np.concatenate(imgs, axis=-1)
    return imgs


def tile_and_augment(image, tile_size=256, num_translations=0, rotations=1, exposure_adjust=None, offset=[0, 0]):
    """Perform data augmentation, including translations, rotations, and exposure adjusmtents.
    Args:
        tiles: array of the number of tiles in the x and y directions
        excess: array of the number of pixels of excess in the x and y directions
        imgdata: array of the image data to be augmented
        offset: array of the number of pixels to offset the tiles in the x and y directions
        divfactor: integer of the number of divisions to make in the x and y directions
        exposure_adjust: boolean of whether to adjust the exposure of the image
    Returns:
        augmented_stack: array of the augmented image data"""
    # Create a list of tiles from the image data, performing translation augmentation as well
    if num_translations > 0:
        aug_stack, tile_shape = augment_translation(image, tile_size=tile_size, num_translations=num_translations) # Gives (xtiles * ytiles * N_trans**2, 256, 256)
    else:
        aug_stack, tile_shape = tileslice(image, tile_size, offset) # will give (xtiles * ytiles, 256, 256)
    # Perform mirroring and rotation augmentation on the translated tiles
    if rotations > 1:
        aug_stack = augment_rot_mirror(aug_stack, rotations=rotations)
    # Perform exposure adjustment augmentation on the augmented tiles
    if exposure_adjust is not None:
        aug_stack =  augment_exposure(aug_stack, num_adjustments=2, adjustment_size=exposure_adjust)
    # print("   -> Created {} tiles that were augmented to produce {} tiles.".format(np.prod(num_tiles), aug_stack.shape[0]))
    return np.array(aug_stack), tile_shape


def augment_translation(image, tile_size=256, num_translations=3):
    """Function for translating the tiles around the image.
    Args:
        image: An array of the image to translate. shape (H, W)
        tile_size: The size of the tiles to translate. int
        num_translations: The number of translations to perform. int or tuple/list/array of ints of shape (2,)
    Returns:
        translated_images: An array of the translated images. shape (num_tran^2 * num_tiles_in_x * num_tiles_in_y, tile_size, tile_size)
                           ex./ 256x256 tiles for 1900x1700 image with 3 translations per axis will return an array of shape (9*7*6=378, 256, 256)"""
    # Get number of tiles and the excess based on the image and tile size
    excess = np.array(image.shape) % tile_size
    if excess[0] == 0:
        offset_chain1 = np.array([0])
    else:
        offset_chain1 = np.linspace(0, excess[0] - 1, num_translations, dtype=int)
    if excess[1] == 0:
        offset_chain2 = np.array([0])
    else:
        offset_chain2 = np.linspace(0, excess[1] - 1, num_translations, dtype=int)
    # Check for edge cases
    if len(offset_chain1) == 1 and len(offset_chain2) == 1:
        print("\tImage is too small to augment translations. Excess: {}".format(excess))
        return tileslice(image, tile_size)
    # Create the augmented images
    augmented_images = []
    count = 0
    for i in offset_chain1:
        for j in offset_chain2:
            output, tile_shape = tileslice(image, tile_size, (i, j))
            augmented_images.extend(output)
            count += output.shape[0]
    augmented_images = np.array(augmented_images)
    return augmented_images, tile_shape


def augment_rot_mirror(images, rotations):
    """Data augmentation by rotating the provided images to create unique representations of the same image.
    Args:
        images: A list of images to augment. shape (N,256,256)
        rotations: The number of rotations to perform, must be > 1. int
    Returns:
        augmented_images: An array of augmented images. shape (N*8,256,256)"""
    # Convert to numpy array
    images = np.array(images)
    augmented_images = np.zeros((images.shape[0] * rotations, images.shape[1], images.shape[2]), dtype=images.dtype)
    # Pack augmented images into augmented_images
    augmented_images[0::rotations] = images
    augmented_images[1::rotations] = np.rot90(images, 1, (1, 2)) # rotate 90
    if rotations > 2: augmented_images[2::rotations] = np.rot90(images, 3, (1, 2)) # rotate -90
    if rotations > 3: augmented_images[3::rotations] = augmented_images[3::rotations] = np.flip(images, axis=1) # flip up/down
    if rotations > 4: augmented_images[4::rotations] = augmented_images[4::rotations] = np.flip(images, axis=2) # flip left/right
    if rotations > 5: augmented_images[5::rotations] = np.flip(augmented_images[4::rotations], axis=1) # flipleft/right + up/down
    if rotations > 6: augmented_images[6::rotations] = np.flip(augmented_images[1::rotations], axis=1) # rotate +90 + flip up/down
    if rotations > 7: augmented_images[7::rotations] = np.flip(augmented_images[2::rotations], axis=1) # rotate -90 + flip up/down
    return augmented_images


def augment_exposure(imgdata, num_adjustments=3, adjustment_size=0.1):
    """Vary the exposure of a image by adjusting the gamma value
    Args:
        imgdata: a list of images to be adjusted
        num_adjustments: the number of different gamma values to use
        adjusmtent_size: the size of the adjustment, must be between 0 and 1
    Returns:
        an array of images with the exposure adjusted"""
    # Get the adjustment values
    adjustments = 1 + adjustment_size * np.arange(-(num_adjustments-1)/2, (num_adjustments-1)/2+1) 
    # Check if the image is a boolean array, if it is, do not adjust the exposure but do create multiple copies
    # Create the output array
    output = np.zeros((imgdata.shape[0] * num_adjustments, imgdata[0].shape[0], imgdata[0].shape[1]), dtype=imgdata.dtype)
    # Loop through the images and adjust gamma
    count = 0
    for i in range(len(imgdata)):
            for adjustment in adjustments:
                if imgdata.dtype != bool:
                    output[count] = exposure.adjust_gamma(imgdata[i], adjustment)
                else:
                    output[count] = imgdata[i]
                count += 1
    return output
