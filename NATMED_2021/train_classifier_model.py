import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler

from CHD_models import create_resnet
from CHD_utils import one_hot, CustomAugmentation, view_schedule


# Data Parameters
image_dim = (80,80,1)
label_map = ["3VT", "3VV", "A4C", "LVOT", "ABDO", "NT"]
# label_map = ["NL", "ABNL"] # for binary classifier
nb_classes = len(label_map)

train_file = "data/TRAIN_80by80.csv"
model_path = "results/model_6view.h5"


# Train Data
X = pd.read_csv(train_file, header=None)
y = np.asarray(X.iloc[:, -5])
y = one_hot(y.astype(np.int), nb_classes)

X = X.iloc[:,:-5].values.reshape(-1,80,80,1).astype(np.float32)
X /= 255.
X_mean = X.mean(axis=0, keepdims=True)
X -= X_mean


# Model Parameters
seed = 7
batch_size = 32
learn_rate = 0.0005
nb_epoch = 175
dropout = 0.5

image_preprocess = CustomAugmentation(rescale_intensity=True,
	gaussian_blur=True)

datagen_args = dict(rotation_range=10,
	width_shift_range=0.20,
	height_shift_range=0.20,
	horizontal_flip=True,
	vertical_flip=True,
	shear_range=0.01,
	zoom_range=0.5,
	fill_mode="constant",
	cval=0.0,
	preprocessing_function=image_preprocess)

image_datagen = ImageDataGenerator(**datagen_args)
schedule = LearningRateScheduler(view_schedule, verbose=0)

# Model Training
model = create_resnet(image_dim, nb_classes, dropout)
adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
# model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"]) # for binary classifier

image_datagen.fit(X, augment=True, seed=seed)
model.fit_generator(image_datagen.flow(X, y, batch_size=batch_size), epochs=nb_epoch,
	steps_per_epoch=np.floor(len(X)/batch_size),
	callbacks=[schedule], verbose=1)

model.save(model_path)
