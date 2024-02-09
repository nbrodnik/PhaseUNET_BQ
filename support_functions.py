import os
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from skimage import exposure, morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import MultipleLocator


#### imaging
def checkimg(imgnp, name="", show=True):
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot()
    ax1.imshow(imgnp,cmap="Spectral")
    ax1.set_title(name)
    if show:
        plt.show()

def show_tiling(im_stack, grid_shape, name="", show=False, cmap='Greys'):
    fig = plt.figure(np.random.randint(1, 100))
    for i in range(len(im_stack)):
        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i+1)
        ax.imshow(im_stack[i], cmap=cmap, vmin=im_stack.min(), vmax=im_stack.max())
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.93, bottom=0.03, left=0.1, right=0.9)
    plt.suptitle(f"Tiled {name} image")
    if show:
        plt.show()

def compare_n_ims(im, titles=None, show=False, grid=None):
    # Compare BSE, Alumina, Output
    fig = plt.figure(4, figsize=(21,7))
    axes = []
    for i in range(len(im)):
        ax = fig.add_subplot(1, len(im), i+1)
        ax.imshow(im[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i])
        if grid is not None:
            ax.xaxis.set_major_locator(MultipleLocator(grid))
            ax.yaxis.set_major_locator(MultipleLocator(grid))
            ax.grid(which="major", axis="both", linestyle="-", color="#CCCCCC")
        axes.append(ax)
    plt.tight_layout()
    if show:
        plt.show()

def checkimg(imgnp,name):
    minmin = np.min(imgnp)
    maxmax = np.max(imgnp)
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot()
    ax1.imshow(imgnp,cmap="Spectral",vmin=minmin,vmax=maxmax)
    ax1.set_title(name)

def compsingletiles(name,*args):
    imgnum = len(args)
    for k in range(imgnum):
        if args[k].ndim > 2:
            if args[k].shape[2]>1:
                args=list(args)
                args[k] = np.expand_dims(args[k][:,:,0],axis=-1)
    fig = plt.figure(figsize=(6*imgnum,6))
    for k in range(imgnum):
        ax = fig.add_subplot(1,imgnum,k+1)
        ax.imshow(args[k],cmap="Spectral")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03, wspace=0.03, top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.suptitle(name)

######### FUNCS

def z_score(volume):
    volume = (volume - volume.mean(axis=(0,1,2), keepdims=True)) / volume.std(axis=(0,1,2), keepdims=True)
    return volume


def tileslice(imginput, tiles, offset=(0,0)):
    """Function for tiling an image into 256x256 tiles.
    Args:
        imginput: np.array of shape (H,W)
        tiles: np.array of length 2, containing x direction number of tiles and y dir number of tiles
        offset: np.array of length 2, containing x and y offset of image from top left corner of image
    Returns:
        np.array of shape (tiles[0]*tiles[1],256,256)"""
    img_stack = []
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            old = (slice(i*256 + offset[0], 256*(i+1) + offset[0]), slice(256*j + offset[1], 256*(j+1) + offset[1]))
            img_stack.append(imginput[old])
    return np.array(img_stack)

def join_tiles(tiles,tilesize,sliceimgnp):
    ##underway
    ind = np.reshape(np.array(range(sliceimgnp.shape[0])),(tiles[0],tiles[1]))
    img = np.zeros((tiles[0]*tilesize,tiles[1]*tilesize))
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            img[i*tilesize:(i+1)*tilesize,j*tilesize:(j+1)*tilesize] = np.squeeze(sliceimgnp[ind[i,j],:,:])
    return img

def loss_def(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()


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
    num_tiles = np.array(image.shape) // tile_size
    # Create a list of tiles from the image data, performing translation augmentation as well
    if num_translations > 0:
        aug_stack = augment_translation(image, tile_size=tile_size, num_translations=num_translations) # Gives (xtiles * ytiles * N_trans**2, 256, 256)
    else:
        aug_stack = tileslice(image, num_tiles, offset) # will give (xtiles * ytiles, 256, 256)
    # Perform mirroring and rotation augmentation on the translated tiles
    if rotations > 1:
        aug_stack = augment_rot_mirror(aug_stack, rotations=rotations)
    # Perform exposure adjustment augmentation on the augmented tiles
    if exposure_adjust is not None:
        aug_stack =  augment_exposure(aug_stack, num_adjustments=2, adjustment_size=exposure_adjust)
    # print("   -> Created {} tiles that were augmented to produce {} tiles.".format(np.prod(num_tiles), aug_stack.shape[0]))
    return np.array(aug_stack)

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
    tiles = np.array(image.shape) // tile_size
    excess = np.array(image.shape) % tile_size
    # Correct number of translations to contain both dimensions
    if type(num_translations) == int:
        num_translations = [num_translations, num_translations]
    # Create the offset chains
    offsetchain1 = np.linspace(0, excess[0] - 1, num_translations[0], dtype=int)
    offsetchain2 = np.linspace(0, excess[1] - 1, num_translations[1], dtype=int)
    # Create the augmented images
    augmented_images = np.zeros((np.prod(tiles) * np.prod(num_translations), tile_size, tile_size), dtype=image.dtype)
    count = 0
    for i in offsetchain1:
        for j in offsetchain2:
            output = np.array(tileslice(image, tiles, [i, j]))
            augmented_images[count: count + output.shape[0]] = output
            count += output.shape[0]
    return augmented_images

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

def rgbstacks(bse_stack_256,al_stack_256):
    bse_out = np.stack((bse_stack_256,bse_stack_256,bse_stack_256),axis=3)
    al_out = np.expand_dims(al_stack_256,axis=-1)
    return bse_out,al_out

def grayscalestacks(bse_stack_256,al_stack_256):
    bse_out = np.expand_dims(bse_stack_256,axis=-1)
    al_out = np.expand_dims(al_stack_256,axis=-1)
    return bse_out,al_out

def interactive_view(image_stack, mask_stack):
    """Create a matplotlib window that allows you to scroll through the image stack and toggle the mask on/off with a button press.
    Args:
        image_stack: An array of images to view. shape (N,256,256)
        mask_stack: An array of masks to view. shape (N,256,256)"""
    # Create the figure
    fig, ax = plt.subplots()
    # Create the image object
    im = ax.imshow(image_stack[0])
    # Create the mask object
    mask = ax.imshow(mask_stack[0], alpha=0.5, cmap='Reds', vmax=1, vmin=0)
    # Create the button
    axbutton = plt.axes([0.7, -0.2, 0.1, 0.075])
    button = Button(axbutton, 'Toggle Mask')
    # Create the slider
    axslider = plt.axes([0.2, -0.05, 0.65, 0.03])
    slider = Slider(axslider, 'Image', 0, image_stack.shape[0]-1, valinit=0, valstep=1)
    # Define the update function
    def update(val):
        im.set_data(image_stack[int(slider.val)])
        mask.set_data(mask_stack[int(slider.val)])
        fig.canvas.draw_idle()
    # Define the button function
    def toggle_mask(event):
        if mask.get_alpha() == 0.5:
            mask.set_alpha(0)
        else:
            mask.set_alpha(0.5)
        fig.canvas.draw_idle()
    # Connect the functions to the objects
    slider.on_changed(update)
    button.on_clicked(toggle_mask)
    # Show the figure
    plt.show()

def combine_tiles(stack, tiling_shape):
    output = np.zeros((tiling_shape[0]*256, tiling_shape[1]*256), dtype=stack.dtype)
    count = 0
    for i in range(tiling_shape[0]):
        for j in range(tiling_shape[1]):
            if j == 0:
                combined = stack[count]
            else:
                combined = np.hstack((combined, stack[count]))
            count += 1
        if i == 0:
            output = combined
        else:
            output = np.vstack((output, combined))
    return output

