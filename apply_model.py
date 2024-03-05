
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
import numpy as np
from skimage import io, exposure
from keras.models import load_model
import tensorflow as tf

import support_functions as supp
import performance_metrics as pm
from gradient_accumulator.GAModelWrapper import GAModelWrapper

# Configure GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

######Get training BSE and AL images
# Define tilesize
tilesize = 256
# Define if uising grayscale images directly (=1) or stacking them to be sudo rgb (=3)
inputdepth = 1
# Number of batches to run
batch_size = 16
# What combining function to use
combining_function = ["mean", "max", "min"][1]
# What axis to run on
clahe = False
# File extension
ext = ".tif"
# Save probabilities?
save_probabilities = False
# Theta threshld (should be pre-determined)
theta = 0.5
# Path to model
model_path = ".../model.h5"
# Path to folder containing the images to run the model on
images_folder = ".../"

#####
model = load_model(model_path, custom_objects={"IoU_Keras": pm.IoU_Keras, "GAModelWrapper": GAModelWrapper})
print("  -> Reading in BSE images")
images = []
masks = []
probabilities = []
paths = []
for p in os.listdir(images_folder):
    if p.endswith(ext):
        paths.append(p)
        bse = io.imread(images_folder + p)
        bse = (bse - bse.min()) / (bse.max() - bse.min())
        images.append(bse)
images = np.array(images)

print("  -> Running model on the BSE images.")
for i in range(images.shape[0]):
    if clahe:
        bse_image = exposure.equalize_adapthist(images[i])
    else:
        bse_image = images[i]
    tiles = np.array(bse_image.shape) // tilesize
    excess = np.array(bse_image.shape) % tilesize
    bse_image_tiled = supp.tile_and_augment(bse_image, tilesize)
    bse_image_tiled = np.expand_dims(bse_image_tiled, axis=-1).astype(np.float32)
    output0 = model.predict(bse_image_tiled, verbose=0, batch_size=batch_size)
    output0 = np.squeeze(supp.combine_tiles(output0, tiles))
    if excess[0] == 0 and excess[1] == 0:
        output = output0
    else:
        bse_image_tiled = supp.tile_and_augment(bse_image, tilesize, offset=excess)
        bse_image_tiled = np.expand_dims(bse_image_tiled, axis=-1).astype(np.float32)
        output1 = model.predict(bse_image_tiled, verbose=0, batch_size=batch_size)
        output1 = np.squeeze(supp.combine_tiles(output1, tiles))
        output = np.zeros((2,) + bse_image.shape, dtype=float)
        output[0, :-excess[0], :-excess[1]] = output0
        output[1, excess[0]:, excess[1]:] += output1
        if combining_function == "mean":
            output = output.mean(axis=0)
        elif combining_function == "max":
            output = output.max(axis=0)
        elif combining_function == "min":
            output = output.min(axis=0)
    probabilities.append(output)
    masks.append(output > theta)

probabilities = np.array(probabilities)
masks = np.array(masks)

print("  -> Saving results")
if save_probabilities:
    for i in range(probabilities.shape[0]):
        save_name = os.path.join(images_folder, paths[i].split(".")[0] + "_prob" + ext)
        io.imsave(save_name, probabilities[i])
for i in range(masks.shape[0]):
    save_name = os.path.join(images_folder, paths[i].split(".")[0] + "_mask" + ext)
    io.imsave(save_name, masks[i].astype(np.uint8) * 255)
