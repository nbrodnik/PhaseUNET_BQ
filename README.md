# BasicUNET
Python implementation of a basic U-Net for semantic segmentation of images.

Currently implemented in Tensorflow/Keras, planning to add PyTorch implementation as well.

The following conda command create a known working environment for running the code.


```
#(Windows Native)
conda create -n unet_test python=3.9 numpy scikit-image matplotlib tqdm cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge
pip install --upgrade pip
pip install "tensorflow<2.11"

#(MacOS - NO GPU)
conda create -n unet_test python=3.9 numpy scikit-image matplotlib tqdm -c conda-forge
pip install --upgrade pip
pip install tensorflow

#(Linux or WSL 2)
#See https://www.tensorflow.org/install/pip#linux_1
```


