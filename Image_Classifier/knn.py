from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from dataset import SimpleDatasetLoader
from imutils import paths

dataset = "Image_Classifier/datasets/animals"
n_neighbors = 1
# declaring dimensions of the preprocessing
newWidth = 32
newHeight = 32
# number of channels of images in the dataset
channels = 3
# number of cores to be used for training
cores = -1 # -1 means use all the available cores for training

print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))

#initialize preprocessor and dataset loader
simpleProcessor = SimplePreprocessor(newWidth, newHeight)
simpleDatasetLoader = SimpleDatasetLoader([simpleProcessor])

# read data and labels
(data, labels) = simpleDatasetLoader.load(imagePaths, verbose=500)

# reshape data matrix into a form where every row consists of a flattened image
data = data.reshape((data.shape[0], newWidth*newHeight*channels))

# encode labels as integers starting from 0
labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# train k-NN classifier based on the raw pixel intensities
print("[INFO] training k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=cores)
model.fit(trainX, trainY)
print("[INFO] Evaluating k-NN classifier...")
print(classification_report(testY, model.predict(testX), target_names=labelEncoder.classes_))