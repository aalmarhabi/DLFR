# DLFR 
This repository is for a different languages font recognition via Raspberry Pi project.

# Requirement
```
google-images-download >= 2.4.2
Keras >= 2.2.4
numpy >= 1.15.2
Pillow >= 5.3.0
pyparsing >= 2.2.2
scikit-learn >= 0.20.0
sklearn >= 0.0
tensorboard >= 1.10.0
tensorflow >= 1.10.1
```

# How to navigate this repository?
You have the three main scripts to run the project. The predict.py used to predict language from given image and both the other file use live-stream video feeds to make a prediction (predictPC for laptops and predictRPi for Raspberry Pi.) The model folder has all the results of the trained models on the datasets. In addition, the visualization and image pre-processing folder shows how to visualize the model and how to manipulate the images. Note the dataset is not included but you can make your own as it will be suggested below.

# There are three different version of datasets used in this project:
Version 1: random images that have a specified text init with a different environment.
    Used open library by hardikvasa called google-images-download. It's a python script that allows you to search and download hundreds of images (https://github.com/hardikvasa/google-images-download)
    You can use different keywords to specify the size, the type, and the images format. 
 
```
      $ for example : $googleimagesdownload --keywords "playground" --limit 20 --color red
```

      
Version 2: random images of the specified language alone with white background.
    Created the dataset using Word document of random words taken from different articles and news. Then use an online converter to convert the PDF to separated images. 

Version 3: a modified version of version 2 using Pillow python library.
    Function 1 to do several transformations on the images, e.g., rotation, flipping position and color changing

# How to run the code?
Almost every python script in this project has parse arguments you can check before running the code. For example to run the project on Raspberry Pi using SNN model (also noting that for SNN use -w 32 -e32 -f 1 and for VGG -w 64 -e 64 -f -1)

```
$ python predict.py --model location/simple_nn1.model --label-bin location/simple_nn_lb1.pickle -w 32 -e 32 -f 1
```

# Website
Please check out the project website, .

# Contribute 
Feel free to fork this repository and make your own changes.



