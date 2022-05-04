#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the PyCon.DE 2018 talk by Jens Leitloff and Felix M. Riese.
PyCon 2018 talk: Satellite data is for everyone: insights into modern remote
sensing research with open data and Python.
License: MIT
"""
import os
from glob import glob

import tensorflow as tf
import tensorflow.python.keras as keras
from image_functions import simple_image_generator
from npu_bridge.npu_init import *
from tensorflow.keras.applications.densenet import DenseNet201 as DenseNet
from tensorflow.keras.applications.vgg16 import VGG16 as VGG
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import backend as K

# variables
path_to_split_datasets = "/home/slobodan/tutorials/sentinel/data/ms/"
use_vgg = True
batch_size = 64

class_indices = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}
num_classes = len(class_indices)

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

# Init Ascend
os.environ["ASCEND_DEVICE_ID"] = "0"
os.environ["JOB_ID"] = "10385"
os.environ["RANK_ID"] = "0"
os.environ["RANK_SIZE"] = "1"

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

# parameters for CNN
dim = 32
input_tensor = Input(shape=(dim, dim, 13))
# introduce a additional layer to get from 13 to 3 input channels
input_tensor = Conv2D(3, (1, 1))(input_tensor)
if use_vgg:
    base_model_imagenet = VGG(
        include_top=False, weights="imagenet", input_shape=(dim, dim, 3)
    )
    base_model = VGG(include_top=False, weights=None, input_tensor=input_tensor)
    for i, layer in enumerate(base_model_imagenet.layers):
        # we must skip input layer, which has no weights
        if i == 0:
            continue
        base_model.layers[i + 1].set_weights(layer.get_weights())
else:
    base_model_imagenet = DenseNet(
        include_top=False, weights="imagenet", input_shape=(dim, dim, 3)
    )
    base_model = DenseNet(include_top=False, weights=None, input_tensor=input_tensor)
    for i, layer in enumerate(base_model_imagenet.layers):
        # we must skip input layer, which has no weights
        if i == 0:
            continue
        base_model.layers[i + 1].set_weights(layer.get_weights())

# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
# top_model = Flatten()(top_model)

# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation="relu")(top_model)
    top_model = Dense(2048, activation="relu")(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation="softmax")(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# print network structure
model.summary()

# defining ImageDataGenerators
# ... initialization for training
training_files = glob(path_to_train + "/**/*.tif")
train_generator = simple_image_generator(
    training_files,
    class_indices,
    batch_size=batch_size,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
)

# ... initialization for validation
validation_files = glob(path_to_validation + "/**/*.tif")
validation_generator = simple_image_generator(
    validation_files, class_indices, batch_size=batch_size
)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
# set convolution block for reducing 13 to 3 layers trainable
for layer in model.layers[:2]:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint(
    "/home/slobodan/tutorials/sentinel/data/models/"
    + file_name
    + "_ms_transfer_init."
    + "{epoch:02d}-{val_categorical_accuracy:.3f}."
    + "hdf5",
    monitor="val_categorical_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
earlystopper = EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=10,
    mode="max",
    restore_best_weights=True,
)
history = model.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=100,
    callbacks=[checkpointer, earlystopper],
    validation_data=validation_generator,
    validation_steps=500,
)
initial_epoch = len(history.history["loss"]) + 1

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
names = []
for i, layer in enumerate(model.layers):
    names.append([i, layer.name, layer.trainable])
print(names)

if use_vgg:
    # we will freaze the first convolutional block and train all
    # remaining blocks, including top layers.
    for layer in model.layers[:2]:
        layer.trainable = True
    for layer in model.layers[2:5]:
        layer.trainable = False
    for layer in model.layers[5:]:
        layer.trainable = True
else:
    for layer in model.layers[:2]:
        layer.trainable = True
    for layer in model.layers[2:8]:
        layer.trainable = False
    for layer in model.layers[8:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint(
    "/home/slobodan/tutorials/sentinel/data/models/"
    + file_name
    + "_ms_transfer_final."
    + "{epoch:02d}-{val_categorical_accuracy:.3f}"
    + ".hdf5",
    monitor="val_categorical_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
earlystopper = EarlyStopping(
    monitor="val_categorical_accuracy", patience=10, mode="max"
)
model.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=100,
    callbacks=[checkpointer, earlystopper],
    validation_data=validation_generator,
    validation_steps=500,
    initial_epoch=initial_epoch,
)

sess.close()
