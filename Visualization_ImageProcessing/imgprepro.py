# import Pillow known as PIL
from PIL import Image, ImageFilter
import os
import numpy as np
import PIL.ImageOps

# this creat image object to work with
# to show image you just use methods .show()
#im1.show()
# save file in different type image we can use .save methods
#im1.save('delete1.png')
# write code that loop in and save jpeg file into png
foldername1 = '/Users/alhussainalmarhabi/OpencvProject/DLFR_VER1/languages/arabic/'
foldername2 = '/Users/alhussainalmarhabi/OpencvProject/DLFR_VER1/languages/english/'
foldername3 = '/Users/alhussainalmarhabi/OpencvProject/DLFR_VER1/languages/japanese/'
# we do loop on length of foldername which is 3
foldername = [foldername1, foldername2, foldername3]

# making function
def normapro(folder, f):
    # note f is now the name of one image
    i = Image.open(folder + f)
    # split the image name and extension
    fn, fext = os.path.splitext(f)
    # do rotation of 45 and crop the image
    out1 = i.rotate(45)
    box = (350, 0, 650, 400)  # box = (30, 30, 410, 410)
    region1 = out1.crop(box)
    region1.save(folder + '{}_prepro1.jpg'.format(fn))
    # do rotation of 90 and crop the image
    # out2 = i.transpose(Image.ROTATE_90)
    # box = (30, 100, 410, 600)
    # region2 = out2.crop(box)
    # region2.save(folder + '{}_prepro2.jpg'.format(fn))
    # flip images  left to right and opposit
    out3 = i.transpose(Image.FLIP_LEFT_RIGHT)
    out3.save(folder + '{}_prepro3.jpg'.format(fn))
    # flip images top to bottom
    out4 = i.transpose(Image.FLIP_TOP_BOTTOM)
    out4.save(folder + '{}_prepro4.jpg'.format(fn))
    # invert black to white and opposit
    out5 = PIL.ImageOps.invert(i)
    out5.save(folder + '{}_prepro5.jpg'.format(fn))
    # change the color of the images
    orig_color = (255, 255, 255)
    replacement_color1 = (200, 0, 0)
    replacement_color2 = (0, 200, 0)
    replacement_color3 = (0, 0, 200)
    img = i.convert('RGB')
    data1 = np.array(img)
    data2 = np.array(img)
    data3 = np.array(img)
    data1[(data1 != orig_color).all(axis=-1)] = replacement_color1
    data2[(data2 != orig_color).all(axis=-1)] = replacement_color2
    data3[(data3 != orig_color).all(axis=-1)] = replacement_color3
    out6 = Image.fromarray(data1, mode='RGB')
    out7 = Image.fromarray(data2, mode='RGB')
    out8 = Image.fromarray(data3, mode='RGB')
    out6.save(folder + '{}_prepro6.jpg'.format(fn))
    out7.save(folder + '{}_prepro7.jpg'.format(fn))
    out8.save(folder + '{}_prepro8.jpg'.format(fn))
    # now incrument the breaker bk

def bnwpro(folder, f):
    # note f is now the name of one image
    i = Image.open(folder + f)
    # split the image name and extension
    fn, fext = os.path.splitext(f)
    #
    inverted_image = PIL.ImageOps.invert(i)
    inverted_image.save(folder + '{}.jpg'.format(fn))

for folder in foldername:
    bk = 0
    for f in os.listdir(folder):
        # bk should be equal the number of cycile you want
        # len(os.listdir(folder)) for entire images
        if bk == 60:
            break
        if f.endswith('.jpg'):
            normapro(folder, f)
        bk += 1
