# reference code to Jason Brownlee - machine learning mastery and Adrian Rosebrock - pyimagesearch
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
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
	image = cv2.resize(image, (64, 64)) # (64,64)
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



# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=35, width_shift_range=0.2,
	height_shift_range=0.30, shear_range=0.15, zoom_range=0.3,
	horizontal_flip=True, fill_mode="reflect") # fill_mode='nearest', wrap, reflect, constant

# create a grid of 3x3 images
plt.figure(0)
for i in range(0, 9):
	plt.subplot(330 + 1 + i)
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

plt.suptitle("Random Batch of Dataset Before DIG")


# Fit the data image generator to the training data
aug.fit(trainX)

# create a grid of 3x3 images
# configure batch size and retrieve one batch of images
plt.figure(1)
for X_batch, y_batch in aug.flow(trainX, trainY, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i], cmap=plt.get_cmap('gray'))
	break

plt.suptitle("Random Batch of Dataset After DIG")
# show the plot
plt.show()








