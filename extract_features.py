from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from data_flow.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import random
import os

datasetPath = "datasets/animals/images"
outputHDF5  = "datasets/animals/hdf5/animals_features.hdf5"
batchSize   = 32
bufferSize  = 1000

print("[INFO] loading images...")
imagesPaths = list(paths.list_images(datasetPath))
random.shuffle(imagesPaths)

# extract the class labels from the image paths then encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagesPaths]
labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)

# initialize the HDF5 dataset writer, then store the class label names 
# in the dataset
dataset = HDF5DatasetWriter((len(imagesPaths), 512 * 7 * 7),
                outputHDF5, dataKey="features", bufSize=bufferSize)
dataset.storeClassLabels(labelEncoder.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
                    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagesPaths), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagesPaths), batchSize):
    # extract the batch of images and labels
    batchPaths  = imagesPaths[i: i + batchSize]
    batchLabels = labels[i: i + batchSize]

    # initialize the list of actual images that will be passed through the network
    # for feature extraction
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by
            # (1) expanding the dimensions and
            # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batchSize)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()