from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
from utils import plotHistory

numEpochs = 100

print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_openml("mnist_784")
data    = dataset["data"]
target  = dataset["target"]

# scale the raw pixel intensities to the range [0, 1.0]
data = data.astype("float") / 255.0

# construct the training and testing splits
trainX, testX, trainY, testY = train_test_split(data, target, test_size=0.25)

# convert the labels from integers to vectors / hot-encoding
labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY = labelBinarizer.transform(testY)

# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
history = model.fit(trainX, trainY, validation_data=(testX, testY), 
                epochs=numEpochs, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
        target_names=[str(x) for x in labelBinarizer.classes_]))

# plot the training loss and accuracy
plotHistory(history, numEpochs)