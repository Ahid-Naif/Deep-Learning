from keras.applications import VGG16

isTopIncluded = True

print("[INFO] loading network...")
model = VGG16(weights="imagenet",   include_top=isTopIncluded)
print("[INFO] showing layers...")

# loop over the layers in the network and display them to the console
for (i, layer) in enumerate(model.layers):
    print("[INFO] " + str(i) + " " + str(layer))