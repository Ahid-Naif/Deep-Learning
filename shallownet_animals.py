from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from preprocessing import ImageToArrayPreprocessor
from dataset import SimpleDatasetLoader
from nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
from utils import plotHistory

datasetPath = "datasets/animals"
newWidth  = 32
newHeight = 32
channels  = 3
classes   = ["cat", "dog", "panda"]
numEpochs = 100

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(datasetPath))

# initialize the image preprocessors
simplePreprocessor    = SimplePreprocessor(newWidth, newHeight)
imageToArrayProcessor = ImageToArrayPreprocessor()

# load the dataset from disk
simpleDatasetLoader = SimpleDatasetLoader(preprocessors=[simplePreprocessor, imageToArrayProcessor])
(data, labels) = simpleDatasetLoader.load(imagePaths, verbose=500)

# scale the raw pixel intensities to the range [0, 1]
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, 
                                    test_size=0.25, random_state=42)

# hot-encoding
trainY = LabelBinarizer().fit_transform(trainY)
testY  = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(0.005)
model = ShallowNet.build(width=newWidth, height=newHeight, depth=channels, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, validation_data=(testX, testY),
                        batch_size=32, epochs=numEpochs, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save("shallownet_weights.hdf5")

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=classes))

plotHistory(history, numEpochs)