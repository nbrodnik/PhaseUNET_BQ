# BasicUNET
Python implementation of a basic U-Net for semantic segmentation of images.

Currently implemented in Tensorflow/Keras, planning to add PyTorch implementation as well.

conda/mamba environment instructions (taken from tensorflow documentation, assumes windows machine)
`conda create -n unet_test python=3.9 numpy scikit-image matplotlib tqdm cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge
pip install --upgrade pip
pip install "tensorflow<2.11"`
