# reference pyimagesearch site for OpenCV

'''
tree --dirsfirst --filelimit 10

 python train_simple_nn.py --dataset languages/ --model output/simple_nn.model
	--label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
'''
# set the matplotlib backend so figures can be saved in the background
import matplotlib
# Renderer AGG (Anti-Grain Geometry) raster graphics type png
matplotlib.use("Agg")

# import the necessary SCIKIT-LEARN packages
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# import the necessary KERAS packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# import the necessary IMUTILS packages
from imutils import paths

# import other necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
# .list_images will find all path to input images in the --dataset
imagePaths = sorted(list(paths.list_images(args["dataset"]))) #"languages/" args["dataset"]
random.seed(42) # random reordering
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring
    # aspect ratio), flatten the image into 32x32x3=3072 pixel image
    # into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    #label = 1 if label == "english" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
# ==standarization== we scale pixel intensities from [0, 255] to [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

'''
So what we did so far is loaded the data after preprocessing and then 
randomly shuffle them. Now, we want to split them to training and testing split
'''
# TRAINING AND TEST SPLITS
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] splitting data ")
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY) # use .fit_transform as reference
# trainY =to_categorical(trainY, num_classes=2)
# testY = to_categorical(testY, num_classes=2) # use .fit_transform as reference

'''
So far we should have the label for different classes languages 
[IN CASE OF TWO LANGUAGES]:
[0,1] => for English
[1,0] => for Arabic
[IN CASE OF FIVE LANGUAGES]:
[1, 0, 0, 0, 0]
[0, 1, 0, 0, 0]
[0, 0, 1, 0, 0]
[0, 0, 0, 1, 0]
[0, 0, 0, 0, 1]
'''

# SETUP KERAS MODEL ARCHITECTURE
# Number of inputs 32*32*3 = 3072
# define the 3072-1024-512-3 architecture using Keras
# model = Sequential()
# model.add(Dense(1024, input_shape=(3072,), activation="relu"))
# model.add(Dense(512, activation="relu"))
# model.add(Dense(len(lb.classes_), activation="softmax")) #len(lb.classes_)
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(len(lb.classes_), activation="softmax")) #len(lb.classes_)


# COMPILE YOUR KERAS MODEL
# for training we need (Model, Optimizer, loss)
# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using Stochastic Gradient Descent
# SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification rather categorical_crossentropy )
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
# opt = Adadelta() import above
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# FIT KERAS MODEL TO THE DATA
#SO FAR: Training Split --> Compiled Model --> Fit Model
# train the neural network
# batch_size is the size of each group of data to pass through the network.
# if using large GPU will take larger batch sizes
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              epochs=EPOCHS,
              batch_size=32)

# Evaluate your keras model
# Fit Model --> Testing Split --> Predictions
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Training Loss and Accuracy (SNN)")
ax1.set_xlabel("Epoch #")
lns1 = ax1.plot(N, H.history["loss"], label="train_loss")
lns2 = ax1.plot(N, H.history["val_loss"], label="validation_loss")
#ax1.legend(loc=0)
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
lns3 = ax2.plot(N, H.history["acc"], label="train_acc", color='k')
lns4 = ax2.plot(N, H.history["val_acc"], label="validation_acc", color='m')
ax2.set_ylabel("Accuracy")
#ax2.legend(loc=0)

# added these three lines
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=7)


fig.tight_layout()
plt.savefig(args["plot"])


# FINAL STEP IS TO SAVE OUR TRAINED MODEL

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()