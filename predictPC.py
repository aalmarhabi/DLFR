'''
how to call the program
python predict.py
-m 'output/simple_nn1.model'
-l 'output/simple_nn_lb1.pickle'
-w 32 -e 32 -f 1
-i 'test_images/arabic.jpg'

'''
# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import argparse
import pickle
import os
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
                help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
                help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
                help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
                help="whether or not we should flatten the image")
args = vars(ap.parse_args())


# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream")
# for pc cam
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    # prepare the image to be classified by our deep learning network
    a = args["width"]
    b = args["height"]
    image = cv2.resize(frame, (a, b))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # check to see if we should flatten the image and add a batch
    # dimension
    if args["flatten"] > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # otherwise, we must be working with a CNN -- don't flatten the
    # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
                               image.shape[2]))


    # make a prediction on the image
    preds = model.predict(image)

    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]


    # draw the class label + probability on the output image
    text = "{}: {:.3f}%".format(label, preds[0][i] * 100)
    # (frame, text, location, font type, font size, font color, font boldness)
    frame = cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2 ,
                (128, 0, 0), 3)

    # show the output image
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # quite the program
    if key == ord("q"):
        break

# close all windows
print("[INFO] closing up")
cv2.destroyAllWindows()
vs.stop()
