import h5py

hdf5FilePath = "datasets/animals/hdf5/animals_features.hdf5"

db = h5py.File(hdf5FilePath)
print(list(db.keys()))

print(db["features"].shape)
print(db["labels"].shape)
print(db["label_names"].shape)