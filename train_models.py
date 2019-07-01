from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import os

outputDirectory = "output"
outputModels    = "models"
numModels       = 5
numEpochs       = 40

# load the training and testing data
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale data into the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY  = labelBinarizer.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                height_shift_range=0.1, horizontal_flip=True,
                fill_mode="nearest")

# loop over the number of models to train
for i in np.arange(0, numModels):
    # initialize the optimizer and model
    print("[INFO] training model " + str(i+1) + "/" + str(numModels))
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                    metrics=["accuracy"])

    # train the network
    history = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                    validation_data=(testX, testY), epochs=numEpochs,
                    steps_per_epoch=len(trainX) // 64, verbose=1)

    # save the model to disk
    p = [outputModels, "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # evaluate the network
    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
                    predictions.argmax(axis=1), target_names=labelNames)

    # save the classification report to file
    p = [outputDirectory, "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()