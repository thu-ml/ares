from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Lambda
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist, cifar10, cifar100
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
import math
from utils.model import resnet_v1, resnet_v2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64, '')
tf.app.flags.DEFINE_float('mean_var', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_string('optimizer', 'mom', '')
tf.app.flags.DEFINE_integer('version', 1, '')
tf.app.flags.DEFINE_float('lr', 0.01, 'initial lr')
tf.app.flags.DEFINE_integer('feature_dim', 256, '')
tf.app.flags.DEFINE_bool('is_2d_demo', False, 'whether is a 2d demo on MNIST')
tf.app.flags.DEFINE_bool('use_ball', True, 'whether use ball loss or softmax')
tf.app.flags.DEFINE_bool('use_MMLDA', True, 'whether use MMLDA or softmax')
tf.app.flags.DEFINE_bool('use_BN', True, 'whether use batch normalization in the network')
tf.app.flags.DEFINE_bool('use_random', False, 'whether use random center or MMLDA center in the network')
tf.app.flags.DEFINE_bool('use_dense', True, 'whether use extra dense layer in the network')
tf.app.flags.DEFINE_bool('use_leaky', False, 'whether use leaky relu in the network')
tf.app.flags.DEFINE_string('dataset', 'mnist', '')



random_seed = '' # '' or '2' or '3'
# Load the dataset
if FLAGS.dataset=='mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.repeat(np.expand_dims(x_train, axis=3), 3, axis=3)
    x_test = np.repeat(np.expand_dims(x_test, axis=3), 3, axis=3)
    epochs = 50
    num_class = 10
    epochs_inter = [30,40]
elif FLAGS.dataset=='cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    epochs = 200
    num_class = 10
    epochs_inter = [100,150]
elif FLAGS.dataset=='cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    epochs = 200
    num_class = 100
    epochs_inter = [100,150]
else:
    print('Unknown dataset')


# Training parameters
subtract_pixel_mean = True
version = FLAGS.version # Model version
n = 5 # n=5 for resnet-32 v1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
    feature_dim = 64
elif version == 2:
    depth = n * 9 + 2
    feature_dim = FLAGS.feature_dim

if FLAGS.use_random==True:
    name_random = '_random'
    random_seed = '2'
else:
    name_random = ''

if FLAGS.use_leaky==True:
    name_leaky = '_withleaky'
else:
    name_leaky = ''

if FLAGS.use_dense==True:
    name_dense = ''
else:
    name_dense = '_nodense'

if FLAGS.is_2d_demo==True:
    is_2d_demo = '_demoMNIST'
else:
    is_2d_demo = ''


#Load centers in MMC
kernel_dict = loadmat('kernel_paras/meanvar1_featuredim'+str(feature_dim)+'_class'+str(num_class)+'.mat')
mean_logits_np = kernel_dict['mean_logits'] #num_class X num_dense
mean_logits = FLAGS.mean_var * tf.constant(mean_logits_np,dtype=tf.float32)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)


def dot_loss(y_true, y_pred):
    return - tf.reduce_sum(y_pred * y_true, axis=-1) #batch_size X 1

#MMLDA prediction function
def MMLDA_layer(x, means=mean_logits, num_class=num_class, use_ball=FLAGS.use_ball):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    if use_ball==True:
        return logits
    else:
        logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True) #Avoid numerical rounding
        logits = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True)) #Avoid numerical rounding
        return logits

def lr_schedule(epoch):
    lr = FLAGS.lr
    if epoch > epochs_inter[1]:
        lr *= 1e-2
    elif epoch > epochs_inter[0]:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model_input = Input(shape=input_shape)

#dim of logtis is batchsize x dim_means
if version == 2:
    original_model,_,_,_,final_features = resnet_v2(input=model_input, depth=depth, num_classes=num_class, num_dims=feature_dim, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)
else:
    original_model,_,_,_,final_features = resnet_v1(input=model_input, depth=depth, num_classes=num_class, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)

if FLAGS.use_BN==True:
    BN_name = '_withBN'
    print('Use BN in the model')
else:
    BN_name = '_noBN'
    print('Do not use BN in the model')

if FLAGS.use_MMLDA==True:
    print('Using MM Training Scheme')
    new_layer = Lambda(MMLDA_layer)
    predictions = new_layer(final_features)
    model = Model(input=model_input, output=predictions)
    use_ball_=''
    train_loss = dot_loss
    if FLAGS.use_ball==False:
        print('Using softmax function (MMLDA)')
        use_ball_='_softmax'
    filepath_dir = 'trained_models/'+FLAGS.dataset+'/resnet32v'+str(version)+'_meanvar'+str(FLAGS.mean_var) \
                                                                +'_'+FLAGS.optimizer \
                                                                +'_lr'+str(FLAGS.lr) \
                                                                +'_batchsize'+str(FLAGS.batch_size) \
                                                                +BN_name+name_leaky+name_dense+name_random+random_seed+use_ball_+is_2d_demo
else:
    print('Using softmax loss')
    model = original_model
    train_loss = keras.losses.categorical_crossentropy
    filepath_dir = 'trained_models/'+FLAGS.dataset+'/resnet32v'+str(version)+'_'+FLAGS.optimizer \
                                                            +'_lr'+str(FLAGS.lr) \
                                                            +'_batchsize'+str(FLAGS.batch_size)+BN_name+name_leaky

if FLAGS.optimizer=='Adam':
    model.compile(
            loss=train_loss,
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=['accuracy'])
elif FLAGS.optimizer=='mom':
    model.compile(
            loss=train_loss,
            optimizer=SGD(lr=lr_schedule(0), momentum=0.9),
            metrics=['accuracy'])
model.summary()


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), filepath_dir)
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_loss', mode='min', verbose=2, save_best_only=False, save_weights_only=True, period=5)

lr_scheduler = LearningRateScheduler(lr_schedule)


callbacks = [checkpoint, lr_scheduler]


# Run training, with data augmentation.
print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # randomly flip images
    horizontal_flip=True)

# Compute quantities required for featurewise normalization
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=2,
    workers=4,
    callbacks=callbacks)
