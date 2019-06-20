from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from utils import plotHistory

numEpochs = 40

# load the training and testing data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale data into the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY = labelBinarizer.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=64, epochs=numEpochs, verbose=1)

# evaluate the network 
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=labelNames))

plotHistory(history, numEpochs)