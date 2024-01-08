# FingerprintVerification

The FingerprintVerification Project is a Siamese neural network, an architecture specializing in similarity-based tasks, that is trained for evaluating and determining the likeness between different fingerprint patterns, which can be used for fingerprint verification between users. (dataset not included)

## Components and functions

In this section, the content of the project will be explained.

### Folder: data

**deleted_files.txt**

Contains names of images excluded from training process.

### Folder: src

**data/make_dataset.py**

Contains script which transforms dataset from raw to processed images.

In summary, script performs:

Removal of white spaces around images. Transforms images by performing dilatation, erosion and binarization. Reshape images to size of 136 x 153 px. Deletes samples based on contents of file deleted_files.txt.

**data/fingerprint_dataset.py**

Contains class definition of dataset creation. Dataset is created by looping through every image, creating both matching and mismatching pairs.
Single element in dataset contains: first image's name, second image's name, binary value of 0 or 1 (if images belong to the same user). 

**models/siamese_nn.py**

Contains definition of the neural network class. Chosen type of neural network is Siamese Neural Network, which operates by creating two instances of autoencoders, that process two images at the same time and send results to equate by loss function.

**models/train_model.py**

Contains definition of ContrastiveLoss class, which calculates loss function for training process and functions responsible for setting up the process of training.

Functions: 

make() - loads data to PyTorch Dataloader and splits it to training and testing sets,

train_and_log() - initiates training and tracks performace using "Weights and Biases" web tracker,

test() - tests trained neural network on samples from testing set,

checkpoint() - saves weights after training,

model_pipeline() - pipeline function that triggers mentioned functions to perform training.

**others/take_scan.py**

Contains functions that supports usage of external fingerprint scanner to gather data and to serve as a verification tool.
