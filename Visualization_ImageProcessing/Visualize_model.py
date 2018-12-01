# import the necessary packages
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
                help="path to trained Keras model")

args = vars(ap.parse_args())


# load the model and label binarizer
print("[INFO] Visualizing the model")
model = load_model(args["model"])

# if you want to like to see summary of the architecture uncomment the line below
#cprint(model.summary())
name = input('Save file as, include extension: ')

plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)


