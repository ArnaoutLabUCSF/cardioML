import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler

from CHD_models import create_unet8
from CHD_utils import vid_schedule, segment_generator, soft_dice


# Data Parameters
image_dim = (272,272,1)
label_dim = (272,272,3)

COLORS_lightred = (125,135,255)
COLORS_darkered = (65,65,255)
COLORS_lightblue = (255,254,103)
COLORS_darkblue = (255,128,0)
color_list = [COLORS_lightred, COLORS_darkered, COLORS_lightblue, COLORS_darkblue, (0,0,0)]
label_map = ["LA", "LV", "RA", "RV", "background"]
nb_classes = len(label_map)

train_images_array = "data/VID_TRAIN_images.npy"
train_labels_array = "data/VID_TRAIN_labels.npy"
model_path = "results/model_4chamber.h5"


# Train Data
X = np.load(train_images_array)
X /= 255.
y = np.load(train_labels_array)


# Model Parameters
seed = 7
batch_size = 2
learn_rate = 0.0001
nb_epoch = 500
dropout_list = [0.25,0.25,0.25,0.25]

datagen_args = dict(rotation_range=25, 
   width_shift_range=0.20, 
   height_shift_range=0.20, 
   horizontal_flip=True,
   vertical_flip=True,
   shear_range=0.05,
   zoom_range=0.15, 
   fill_mode="constant",
   cval=0.0)  

image_datagen = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)
schedule = LearningRateScheduler(vid_schedule, verbose=0)


# Model Training
model = create_unet8(image_dim, nb_classes, dropout_list)
adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=[metrics.categorical_accuracy, soft_dice])

image_datagen.fit(X, augment=True, seed=seed)
image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)

mask_datagen.fit(y, augment=True, seed=seed)
mask_generator = mask_datagen.flow(y, batch_size=batch_size, seed=seed)

train_generator = segment_generator(image_generator, mask_generator, color_list)
model.fit_generator(train_generator, epochs=nb_epoch, 
   steps_per_epoch=np.floor(len(X)/batch_size), 
   callbacks=[schedule], verbose=1)

model.save(model_path)
