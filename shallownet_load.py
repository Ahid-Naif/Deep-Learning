from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from dataset import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2

modelPath   = "shallownet_weights.hdf5"
datasetPath = "datasets/animals"

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab 10 of images from the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(datasetPath)))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
simplePreprocessor       = SimplePreprocessor(32, 32)
imageToArrayPreprocessor = ImageToArrayPreprocessor()

# load the dataset from disk 
simpleDatasetLoader = SimpleDatasetLoader(preprocessors=[simplePreprocessor
                                                , imageToArrayPreprocessor])
(data, labels) = simpleDatasetLoader.load(imagePaths)

# scale the raw pixel intensities to the range [0, 1]
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(modelPath)

# make predictions on the images
print("[INFO] predicting...")
predections = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[predections[i]]),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)