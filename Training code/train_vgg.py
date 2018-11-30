# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from smallvggnet import SmallVGGNet
from vggnet import VGGNet
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
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
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY) # use .fit_transform as reference
# # no need to use to_categorical function for more than two clases
# trainY =to_categorical(trainY, num_classes=2)
# testY = to_categorical(testY, num_classes=2) # use .fit_transform as reference


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=35, width_shift_range=0.2,
	height_shift_range=0.30, shear_range=0.15, zoom_range=0.3,
	horizontal_flip=True, fill_mode="reflect") # fill_mode='nearest', wrap, reflect, constant

# depth represent the number of Classes
# initialize our VGG-like Convolutional Neural Network
# Note: change between SmallVGGNet and VGGNet depend on which model to use
model = SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))


# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#opt = Adadelta(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Training Loss and Accuracy (CNN)")
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

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()










