from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import h5py

hdf5FilePath    = "datasets/animals/hdf5/animals_features.hdf5"
outputModelPath = "animals.cpickle"
numJobs = -1

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(hdf5FilePath)

trainPer = 0.75 # training data percentage
i = int(db["labels"].shape[0] * trainPer)
trainData     = db["features"][:i]
trainLabels   = db["labels"][:i]
testingData   = db["features"][i:]
testingLabels = db["labels"][i:]

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=numJobs)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(testingData)
print(classification_report(testingLabels, preds, target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(outputModelPath, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()