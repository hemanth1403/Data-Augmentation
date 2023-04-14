from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image

datagen = ImageDataGenerator(rotation_range=40, shear_range=0.2,
                             fill_mode="nearest")  # , cval=255, horizontal_flip=True, vertical_flip=True
ls = []
img_dir = "Images/"
# size = 128  # 128*128 size
img = os.listdir(img_dir)
for i, img_name in enumerate(img):
    if (img_name.split('.')[1] == "jpg"):
        image = io.imread(img_dir+img_name)
        image = Image.fromarray(image, "RGB")
        # image = image.resize((size, size))
        ls.append(np.array(image))
# print(np.shape(ls[0]))

x = np.array(ls)
# print(np.shape(x))
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir="Augmented", save_prefix="img", save_format='jpg'):
    i += 1
    if (i > 105):
        break
