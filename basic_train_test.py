import os
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import keras.optimizers as ko
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import load_model

import support as supp
from model import keras_basic_unet
import loss_functions as lf
import performance_metrics as pm
from gradient_accumulator.GAModelWrapper import GAModelWrapper

# Configure the GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

########################
###### USER INPUT ######
# name for run
name = "Test"
# Data folder and identifiers
folder = "./demo_data/"
mask_flag = "mask"
data_flag = "data"
ext = ".tif"
# trainingsplit, % data to use for training (1-% is testing)
train_size = 0.8
# Validation split, % of training data to use for validation
validation_split = 0.2
# Define tilesize
tilesize = 256
# Define if uising grayscale images directly (=1) or stacking them to be rgb (=3)
inputdepth = 1
# training epochs for model
epochs = 2
# Number of batches to run
batch_size = 16

# Define if we are training the model or reading in previously trained weights
TRAIN = True
load_from = None

### More advanced inputs
# Optimizer
optimizer = ko.Adam(beta_1=0.99)
# Gradient accumulation parameters (total batch size = batch_size * Grad_Accum_Steps)
Grad_Accum_Steps = 0
eps = 1e-3
clip_factor = 1e-2
# Activation function
activation = "sigmoid"
# loss function
loss_names = ["DSC", "DSC_BCE", "IoU", "Focal1", "Tversky57", "FocalTversky57", "Combo65", "BFCEG1", "BCE"]
loss_index = -2

### Training callbacks and metrics
# kwargs for training callbacks
reduce_lr_rate_kwargs = dict(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
checkpoint_kwargs = dict(verbose=1, save_best_only=True, save_format="tf")
early_stopping_kwargs = dict(patience=10, restore_best_weights=True, verbose=1)

#### END USER INPUT ####
########################


# Get data
x_paths, y_paths = supp.get_paths(folder, ext, data_flag, mask_flag)

y_train, x_train, y_test, x_test, tile_shape = supp.prep_input_data(x_paths, y_paths, train_size=train_size, clahe=False, seed=1)
print("Tile shape: {}".format(tile_shape))
print("Number of training/validation images: {}".format(x_train.shape[0]))
print("Number of testing images: {}".format(x_test.shape[0]))

loss_functions = [lf.DiceLoss,
                  lf.DiceBCELoss,
                  lf.IoULoss,
                  lf.FocalLoss,
                  lf.TverskyLoss,
                  lf.FocalTverskyLoss,
                  lf.ComboLoss,
                  tf.keras.losses.BinaryFocalCrossentropy(gamma=1, from_logits=False),
                  tf.keras.losses.BinaryCrossentropy()]
loss_function = loss_functions[loss_index]

model_name = name + loss_names[loss_index]

# Create or read in the model
if load_from is not None:
    model = load_model(load_from,
                       custom_objects={"IoU_Keras": pm.IoU_Keras,
                                       "IoULoss": lf.IoULoss,
                                       "DiceLoss": lf.DiceLoss,
                                       "DiceBCELoss": lf.DiceBCELoss,
                                       "FocalLoss": lf.FocalLoss,
                                       "TverskyLoss": lf.TverskyLoss,
                                       "FocalTverskyLoss": lf.FocalTverskyLoss,
                                       "ComboLoss": lf.ComboLoss,
                                       "GAModelWrapper": GAModelWrapper})
else:
    input_img = Input((tilesize, tilesize, inputdepth))
    model = keras_basic_unet.get_unet(input_img,
                                      n_filters=32,
                                      dropout=0.05,
                                      batchnorm=True,
                                      activation=activation)
    if Grad_Accum_Steps > 0:
        model = GAModelWrapper(accum_steps=Grad_Accum_Steps,
                               use_agc=True,
                               clip_factor=clip_factor,
                               eps=eps,
                               inputs=model.input,
                               outputs=model.output)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=[pm.IoU_Keras])

if TRAIN:
    # Setup training callbacks
    callbacks = [
        ReduceLROnPlateau(**reduce_lr_rate_kwargs),
        ModelCheckpoint(os.path.join(f'{model_name}.h5'), **checkpoint_kwargs),
        EarlyStopping(**early_stopping_kwargs)
        ]
    # Run the model
    fit_kwargs = dict(batch_size=batch_size,
                      epochs=epochs,
                      validation_split=validation_split,
                      callbacks=callbacks)
    res = model.fit(x_train, y_train, **fit_kwargs)
    # Save the output
    np.save(os.path.join(folder, model_name + "_history.npy"), res.history)
    tf.keras.backend.clear_session()

# Test the model
supp.test(model, y_test, x_test, tile_shape, model_name, batch_size, tilesize)

print("Complete, model and results saved.")
