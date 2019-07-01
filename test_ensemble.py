from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import glob
import os

modelsPath = ""

# load the testing data, then scale it into the range [0, 1]
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

# convert the labels from integers to vectors
labelBinarizer = LabelBinarizer()
testY = labelBinarizer.fit_transform(testY)

# construct the path used to collect the models then initialize the models list
modelsPaths = os.path.sep.join([modelsPath, "*.model"])
modelsPaths = list(glob.glob(modelsPaths))
models = []

# loop over the model paths, loading the model, and adding it to
# the list of models
for (i, modelPath) in enumerate(modelsPaths):
    print("[INFO] loading model " + str(i) + "/" + str(len(modelsPaths)))
    models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
    # use the current model to make predictions on the testing data,
    # then store these predictions in the aggregate predictions list
    predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all model predictions, then show
# a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))