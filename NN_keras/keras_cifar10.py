from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
from utils import plotHistory

# load the training and testing data
print("[INFO] loading CIFAR-10 data...")
( (trainX, trainY), (testX, testY) ) = cifar10.load_data()

# scale it into the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX  = testX.astype("float") / 255.0

# get dimensions of images in the dataset
height  = trainX.shape[1]
width   = trainX.shape[2]
channel = trainX.shape[3]
print(height.shape)

# reshape the data matrix so that each row represents an image
trainX = trainX.reshape((trainX.shape[0], height*width*channel))
testX  = testX.reshape((testX.shape[0], height*width*channel))

# convert the labels from integers to vectors
labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY  = labelBinarizer.transform(testY)

# initialize the label names for the dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(height*width*channel,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
                metrics=["accuracy"])
history = model.fit(trainX, trainY, validation_data=(testX, testY),
                    epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names= labelNames))

# plot the training loss and accuracy
plotHistory(history)