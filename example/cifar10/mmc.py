''' This file provides a wrapper class for MMC (https://github.com/P2333/Max-Mahalanobis-Training) model for CIFAR-10 dataset. '''

import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Input, Lambda
from keras.regularizers import l2
from keras.models import Model
from scipy.io import loadmat

from ares.model import ClassifierWithLogits
from ares.utils import get_res_path, download_res

MODEL_PATH = get_res_path('./cifar10/mmc')


def load(session):
    model = MMC()
    model.load(session, MODEL_PATH)
    return model


def download(model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    h5_name = 'model.180.h5'
    h5_path = os.path.join(model_path, h5_name)
    if not os.path.exists(h5_path):
        url = 'http://ml.cs.tsinghua.edu.cn/~tianyu/MMC/pretrained_models/MMC_mom_advtrain_cifar10/' + h5_name
        download_res(url, h5_path)
    npy_path = os.path.join(model_path, 'mean.npy')
    if not os.path.exists(npy_path):
        download_res('http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/adp_datamean.npy', npy_path)
    mat_path = os.path.join(model_path, 'meanvar.mat')
    if not os.path.exists(mat_path):
        download_res('https://github.com/P2333/Max-Mahalanobis-Training/raw/master/kernel_paras/meanvar1_featuredim256_class10.mat', mat_path)


class MMC(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 10, 0.0, 1.0, (32, 32, 3), tf.float32, tf.int32)

    def load(self, session, model_path):
        keras.backend.set_session(session)
        model_input = Input(shape=self.x_shape)
        _,_,_,_,final_features = resnet_v2(input=model_input, depth=47, num_classes=10, \
                                                    use_BN=True, use_dense=True, use_leaky=False)
        new_layer = Lambda(MMLDA_layer)
        predictions = new_layer(final_features)
        model = Model(input=model_input, output=predictions)

        h5_path = os.path.join(model_path, 'model.180.h5')
        npy_path = os.path.join(model_path, 'mean.npy')
        model.load_weights(h5_path)
        self._model = model
        self._train_mean = np.load(npy_path)

    def _logits_and_labels(self, xs):
        logits = self._model(xs-self._train_mean)
        predicted_labels = tf.cast(tf.argmax(logits, 1), tf.int32)
        return logits, predicted_labels


def MMLDA_layer(x, num_class=10):
    kernel_dict = loadmat(os.path.join(MODEL_PATH, 'meanvar.mat'))
    mean_logits = kernel_dict['mean_logits'] #num_class X num_dense
    means = 10 * tf.constant(mean_logits, dtype=tf.float32)

    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    return logits


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input, depth, num_classes=10, num_dims=256, use_BN=True, use_dense=True, use_leaky=False):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = input
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True, batch_normalization=use_BN)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = use_BN
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_in, conv_first=False, batch_normalization=use_BN)
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False, batch_normalization=use_BN)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if use_BN:
        x = BatchNormalization()(x)

    if use_leaky==True:
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = Activation('relu')(x)
        
    x = GlobalAveragePooling2D()(x)
    #final_features = Flatten()(x)
    if use_dense==True:
        final_features = Dense(
            num_dims, kernel_initializer='he_normal')(x)
    else:
        final_features = x
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features


if __name__ == '__main__':
    download(MODEL_PATH)
