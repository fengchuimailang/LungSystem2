# 参考 https://github.com/carmel-nzhinusoft/implement-ResNeXt-with-keras/blob/master/resnext50-with-keras.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen, urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from utils import *
import time
from tensorflow.keras.models import load_model
from sklearn.datasets import load_files
# from keras.utils import np_utils  For keras > 2.0, please use from keras.utils import to_categorical instead.
from tensorflow.keras.utils import to_categorical
from glob import glob
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Lambda,Concatenate,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import glorot_uniform

model_name = "resNeXt50_ct"
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
batch_size = 8
log_dir = "./logs/" + model_name + time_str
work_dir = "/home/liubo/nn_project/LungSystem2/workdir/" + model_name + time_str
model_save_path = "/home/liubo/nn_project/LungSystem2/models/guaduate/" + model_name
learn_rate = 0.0001


def split(inputs, cardinality):
    inputs_channels = inputs.shape[3]
    group_size = inputs_channels // cardinality
    groups = list()
    for number in range(1, cardinality + 1):
        begin = int((number - 1) * group_size)
        end = int(number * group_size)
        block = Lambda(lambda x: x[:, :, :, begin:end])(inputs)
        groups.append(block)
    return groups


def transform(groups, filters, strides, stage, block):
    f1, f2 = filters
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))

    transformed_tensor = list()
    i = 1

    for inputs in groups:
        # first conv of the transformation phase
        x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding="valid",
                   name=conv_name + '1a_split' + str(i), kernel_initializer=glorot_uniform(seed=0))(inputs)
        x = BatchNormalization(axis=3, name=bn_name + '1a_split' + str(i))(x)
        x = Activation('relu')(x)

        # second conv of the transformation phase
        x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding="same",
                   name=conv_name + '1b_split' + str(i), kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name + '1b_split' + str(i))(x)
        x = Activation('relu')(x)

        # Add x to transformed tensor list
        transformed_tensor.append(x)
        i += 1

    # Concatenate all tensor from each group
    x = Concatenate(name='concat' + str(stage) + '' + block)(transformed_tensor)

    return x


def transition(inputs, filters, stage, block):
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name='conv2d-trans' + str(stage) + '' + block, kernel_initializer=glorot_uniform(seed=0))(inputs)
    x = BatchNormalization(axis=3, name='batchnorm-trans' + str(stage) + '' + block)(x)
    x = Activation('relu')(x)

    return x


def identity_block(inputs, filters, cardinality, stage, block, strides=(1, 1)):
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))

    # save the input tensor value
    x_shortcut = inputs
    x = inputs

    f1, f2, f3 = filters

    # divide input channels into groups. The number of groups is define by cardinality param
    groups = split(inputs=x, cardinality=cardinality)

    # transform each group by doing a set of convolutions and concat the results
    f1 = int(f1 / cardinality)
    f2 = int(f2 / cardinality)
    x = transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)

    # make a transition by doing 1x1 conv
    x = transition(inputs=x, filters=f3, stage=stage, block=block)

    # Last step of the identity block, shortcut concatenation
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def downsampling(inputs, filters, cardinality, strides, stage, block):
    # useful variables
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))

    # Retrieve filters for each layer
    f1, f2, f3 = filters

    # save the input tensor value
    x_shortcut = inputs
    x = inputs

    # divide input channels into groups. The number of groups is define by cardinality param
    groups = split(inputs=x, cardinality=cardinality)

    # transform each group by doing a set of convolutions and concat the results
    f1 = int(f1 / cardinality)
    f2 = int(f2 / cardinality)
    x = transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)

    # make a transition by doing 1x1 conv
    x = transition(inputs=x, filters=f3, stage=stage, block=block)

    # Projection Shortcut to match dimensions
    x_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides, padding="valid",
                        name='{base}2'.format(base=conv_name), kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name='{base}2'.format(base=bn_name))(x_shortcut)

    # Add x and x_shortcut
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def ResNeXt50(input_shape=(512, 512, 1), classes=5):
    # Transform input to a tensor of shape input_shape
    x_input = Input(input_shape)

    # Add zero padding
    x = ZeroPadding2D((3, 3))(x_input)

    # Initial Stage. Let's say stage 1
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
               name='conv2d_1', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='batchnorm_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = downsampling(inputs=x, filters=(128, 128, 256), cardinality=4, strides=(2, 2), stage=2, block="a")
    x = identity_block(inputs=x, filters=(128, 128, 256), cardinality=4, stage=2, block="b")
    x = identity_block(inputs=x, filters=(128, 128, 256), cardinality=4, stage=2, block="c")

    # Stage 3
    x = downsampling(inputs=x, filters=(256, 256, 512), cardinality=4, strides=(2, 2), stage=3, block="a")
    x = identity_block(inputs=x, filters=(256, 256, 512), cardinality=4, stage=3, block="b")
    x = identity_block(inputs=x, filters=(256, 256, 512), cardinality=4, stage=3, block="c")
    x = identity_block(inputs=x, filters=(256, 256, 512), cardinality=4, stage=3, block="d")

    # Stage 4
    x = downsampling(inputs=x, filters=(512, 512, 1024), cardinality=4, strides=(2, 2), stage=4, block="a")
    x = identity_block(inputs=x, filters=(512, 512, 1024), cardinality=4, stage=4, block="b")
    x = identity_block(inputs=x, filters=(512, 512, 1024), cardinality=4, stage=4, block="c")
    x = identity_block(inputs=x, filters=(512, 512, 1024), cardinality=4, stage=4, block="d")
    x = identity_block(inputs=x, filters=(512, 512, 1024), cardinality=4, stage=4, block="e")
    x = identity_block(inputs=x, filters=(512, 512, 1024), cardinality=4, stage=4, block="f")

    # Stage 5
    x = downsampling(inputs=x, filters=(1024, 1024, 2048), cardinality=4, strides=(2, 2), stage=5, block="a")
    x = identity_block(inputs=x, filters=(1024, 1024, 2048), cardinality=4, stage=5, block="b")
    x = identity_block(inputs=x, filters=(1024, 1024, 2048), cardinality=4, stage=5, block="c")

    # Average pooling
    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)

    # Output layer
    x = Flatten()(x)
    x = Dense(classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0),
              name="fc{cls}".format(cls=str(classes)))(x)

    # Create the model
    model = Model(inputs=x_input, outputs=x, name=model_name)

    return model


def train():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset_ct()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 5).T
    Y_test = convert_to_one_hot(Y_test_orig, 5).T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model = ResNeXt50()
    adam = Adam(lr=learn_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    checkpoint = ModelCheckpoint(filepath=work_dir + "/" + model_name + "_" + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    best_model_path = work_dir + "/" + model_name + "_best.hd5"
    checkpoint_fixed_name = ModelCheckpoint(filepath=best_model_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
    model.fit(x=X_train,
              y=Y_train,
              batch_size=8,
              epochs=20,
              validation_data=(X_test, Y_test),
              callbacks=[checkpoint,
                         checkpoint_fixed_name,
                         TensorBoard(log_dir=log_dir)]
              )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    train()
    best_model_path = work_dir + "/" + model_name + "_best.hd5"
    shutil.copy(best_model_path, model_save_path)

