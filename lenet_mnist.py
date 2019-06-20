from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nn.conv import LeNet
from keras.optimizers import SGD
from sklearn import datasets
import numpy as np
from utils import plotHistory

numEpochs = 20

print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_openml("mnist_784")
data    = dataset["data"]
target  = dataset["target"]

# matrix shape should be: num_samples x rows x columns x depth
data = data.reshape(data.shape[0], 28, 28, 1)

# scale the raw pixel intensities to the range [0, 1.0]
data = data.astype("float") / 255.0

# construct the training and testing splits
trainX, testX, trainY, testY = train_test_split(data, target.astype("int"), 
                                    test_size=0.25)

# convert the labels from integers to vectors / hot-encoding
labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY = labelBinarizer.transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, validation_data=(testX, testY),
                        batch_size=128, epochs=numEpochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=[str(x) for x in labelBinarizer.classes_]))

plotHistory(history, numEpochs)