import numpy as np 
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data   = []
        labels = []

        # loop over images in the image path
        for (i, imagePath) in enumerate(imagePaths):
            # load the image
            image = cv2.imread(imagePath)
            # extract the image label assuming the path format 
            # as: dataset_name/<class>/<image>.jpg
            label = imagePath.split(os.path.sep)[-2]

            # Do the preprocessing
            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            # show an update every verbose images
            if (verbose > 0) and (i > 0) and (i + 1)%verbose == 0:
                print("[INFO] processed " + str(i+1) + "/" + str(len(imagePaths)))
        
        return (np.array(data), np.array(labels))