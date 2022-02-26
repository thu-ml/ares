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
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
from utils.keras_wraper_ensemble import KerasModelWrapper
from utils.utils_model_eval import model_eval_targetacc
from sklearn.metrics import roc_auc_score

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50, 'batch_size for attack')
tf.app.flags.DEFINE_string('optimizer', 'mom', '')
tf.app.flags.DEFINE_float('mean_var', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_string('attack_method', 'FastGradientMethod', '')
tf.app.flags.DEFINE_string('attack_method_for_advtrain', 'FastGradientMethod', '')
tf.app.flags.DEFINE_integer('version', 2, '')
tf.app.flags.DEFINE_float('lr', 0.01, 'initial lr')
tf.app.flags.DEFINE_bool('target', True, 'is target attack or not')
tf.app.flags.DEFINE_bool('use_target', False, 'whether use target attack or untarget attack for adversarial training')
tf.app.flags.DEFINE_integer('num_iter', 10, '')
tf.app.flags.DEFINE_bool('use_ball', True, 'whether use ball loss or softmax')
tf.app.flags.DEFINE_bool('use_MMLDA', True, 'whether use MMLDA or softmax')
tf.app.flags.DEFINE_bool('use_advtrain', True, 'whether use advtraining or normal training')
tf.app.flags.DEFINE_float('adv_ratio', 1.0, 'the ratio of adversarial examples in each mini-batch')
tf.app.flags.DEFINE_integer('epoch', 1, 'the epoch of model to load')
tf.app.flags.DEFINE_bool('use_BN', True, 'whether use batch normalization in the network')
tf.app.flags.DEFINE_string('dataset', 'mnist', '')
tf.app.flags.DEFINE_bool('normalize_output_for_ball', True, 'whether apply softmax in the inference phase')
tf.app.flags.DEFINE_bool('use_random', False, 'whether use random center or MMLDA center in the network')
tf.app.flags.DEFINE_bool('use_dense', True, 'whether use extra dense layer in the network')
tf.app.flags.DEFINE_bool('use_leaky', False, 'whether use leaky relu in the network')

# For calculate AUC-scores
tf.app.flags.DEFINE_bool('is_calculate_auc', False, 'whether to calculate auc scores')
tf.app.flags.DEFINE_bool('is_auc_metric_softmax_for_MMC', False, 'whether use softmax to calculate auc metrics for MMC')

# Load the dataset
if FLAGS.dataset=='mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    epochs = 50
    num_class = 10
    epochs_inter = [30,40]
    x_place = tf.placeholder(tf.float32, shape=(None, 28, 28, 3))

elif FLAGS.dataset=='cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    epochs = 200
    num_class = 10
    epochs_inter = [100,150]
    x_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

elif FLAGS.dataset=='cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    epochs = 200
    num_class = 100
    epochs_inter = [100,150]
    x_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

else:
    print('Unknown dataset')


# These parameters are usually fixed
subtract_pixel_mean = True
version = FLAGS.version # Model version
n = 5 # n=5 for resnet-32 v1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
    feature_dim = 64
elif version == 2:
    depth = n * 9 + 2
    feature_dim = 256


if FLAGS.use_random==True:
    name_random = '_random'
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

    
#Load means in MMLDA
kernel_dict = loadmat('kernel_paras/meanvar1_featuredim'+str(feature_dim)+'_class'+str(num_class)+name_random+'.mat')
mean_logits = kernel_dict['mean_logits'] #num_class X num_dense
mean_logits = FLAGS.mean_var * tf.constant(mean_logits,dtype=tf.float32)


#MMLDA prediction function
def MMLDA_layer(x, means=mean_logits, num_class=num_class, use_ball=FLAGS.use_ball):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    if use_ball==True:
        if FLAGS.normalize_output_for_ball==False:
            return logits
        else:
            return tf.nn.softmax(logits, axis=-1)
    else:
        return tf.nn.softmax(logits, axis=-1)


# Load the data.
y_test_target = np.zeros_like(y_test)
for i in range(y_test.shape[0]):
    l = np.random.randint(num_class)
    while l == y_test[i][0]:
        l = np.random.randint(num_class)
    y_test_target[i][0] = l
print('Finish crafting y_test_target!!!!!!!!!!!')

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

clip_min = 0.0
clip_max = 1.0
# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= np.max(x_train_mean)
    clip_max -= np.min(x_train_mean)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)
y_test_target = keras.utils.to_categorical(y_test_target, num_class)


# Define input TF placeholder
y_place = tf.placeholder(tf.float32, shape=(None, num_class))
y_target = tf.placeholder(tf.float32, shape=(None, num_class))
sess = tf.Session()
keras.backend.set_session(sess)


model_input = Input(shape=input_shape)

#dim of logtis is batchsize x dim_means
if version == 2:
    original_model,_,_,_,final_features = resnet_v2(input=model_input, depth=depth, num_classes=num_class, \
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


#Whether use target attack for adversarial training
if FLAGS.use_target==False:
    is_target = ''
else:
    is_target = 'target'


if FLAGS.use_advtrain==True:
    dirr = 'advtrained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain = '_'+is_target+FLAGS.attack_method_for_advtrain
    adv_ratio_name = '_advratio'+str(FLAGS.adv_ratio)
    mean_var = int(FLAGS.mean_var)
else:
    dirr = 'trained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain = ''
    adv_ratio_name = ''
    mean_var = FLAGS.mean_var


if FLAGS.use_MMLDA==True:
    print('Using MMLDA')
    new_layer = Lambda(MMLDA_layer)
    predictions = new_layer(final_features)
    model = Model(input=model_input, output=predictions)
    use_ball_=''
    if FLAGS.use_ball==False:
        print('Using softmax function')
        use_ball_='_softmax'
    filepath_dir = dirr+'resnet32v'+str(version)+'_meanvar'+str(mean_var) \
                                                                +'_'+FLAGS.optimizer \
                                                                +'_lr'+str(FLAGS.lr) \
                                                                +'_batchsize'+str(FLAGS.batch_size) \
                                                                +attack_method_for_advtrain+adv_ratio_name+BN_name+name_leaky+name_dense+name_random+use_ball_+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'
else:
    print('Using softmax loss')
    model = original_model
    filepath_dir = dirr+'resnet32v'+str(version)+'_'+FLAGS.optimizer \
                                                            +'_lr'+str(FLAGS.lr) \
                                                            +'_batchsize'+str(FLAGS.batch_size)+attack_method_for_advtrain+adv_ratio_name+BN_name+name_leaky+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'


wrap_ensemble = KerasModelWrapper(model, num_class=num_class)


model.load_weights(filepath_dir)


# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap_ensemble)
elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.FastGradientMethod(wrap_ensemble)
elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att = attacks.MomentumIterativeMethod(wrap_ensemble)
elif FLAGS.attack_method == 'BasicIterativeMethod':
    att = attacks.BasicIterativeMethod(wrap_ensemble)


# Consider the attack to be constant
eval_par = {'batch_size': FLAGS.batch_size}


if FLAGS.is_calculate_auc:
    # Calculate model preds for clean inputs
    avg_score_nor = np.array([])
    nor_indicator = np.ones((1000,), dtype=int)
    for i in range(10):
        avg_score_nor_batch = sess.run(tf.reduce_max(model(x_place),axis=-1), feed_dict={x_place:x_test[i*100:(i+1)*100]})
        avg_score_nor = np.concatenate((avg_score_nor, avg_score_nor_batch), axis=0)
        print('Calculate score for nor images with batch', i)

    # Calculate model preds for adv inputs
    eps_ = 8 / 256.0
    if FLAGS.target==False:
        y_target = None
    if FLAGS.attack_method == 'FastGradientMethod':
        att_params = {'eps': eps_,
                       'clip_min': clip_min,
                       'clip_max': clip_max,
                       'y_target': y_target}
    else:
        att_params = {'eps': eps_,
                        #'eps_iter': eps_*1.0/FLAGS.num_iter,
                        #'eps_iter': 3.*eps_/FLAGS.num_iter,
                        'eps_iter': 2. / 256.,
                       'clip_min': clip_min,
                       'clip_max': clip_max,
                       'nb_iter': FLAGS.num_iter,
                       'y_target': y_target}
    adv_x = tf.stop_gradient(att.generate(x_place, **att_params))
    preds = tf.reduce_max(model(adv_x),axis=-1)

    if FLAGS.is_auc_metric_softmax_for_MMC==True:
        preds = tf.reduce_max(tf.nn.softmax(model(adv_x)),axis=-1)

    avg_score_adv = np.array([])
    adv_indicator = np.zeros((1000,), dtype=int)
    if FLAGS.target==True:
        for i in range(10):
            avg_score_adv_batch = sess.run(preds, feed_dict={x_place:x_test[i*100:(i+1)*100], y_target:y_test_target[i*100:(i+1)*100]})
            avg_score_adv = np.concatenate((avg_score_adv, avg_score_adv_batch), axis=0)
            print('Calculate score for target attack images with batch', i)
    else:
        for i in range(10):
            avg_score_adv_batch = sess.run(preds, feed_dict={x_place:x_test[i*100:(i+1)*100]})
            avg_score_adv = np.concatenate((avg_score_adv, avg_score_adv_batch), axis=0)
            print('Calculate score for untarget attack images with batch', i)

    score_all = np.concatenate((avg_score_nor,avg_score_adv), axis=0)
    indicator_all = np.concatenate((nor_indicator,adv_indicator), axis=0)
    print('AUC score is', roc_auc_score(indicator_all, score_all))

else:
    for eps in range(4):
        eps_ = (eps+1) * 8
        print('eps is %d'%eps_)
        eps_ = eps_ / 256.0
        if FLAGS.target==False:
            y_target = None
        if FLAGS.attack_method == 'FastGradientMethod':
            att_params = {'eps': eps_,
                       'clip_min': clip_min,
                       'clip_max': clip_max,
                       'y_target': y_target}
        else:
            att_params = {'eps': eps_,
                        #'eps_iter': eps_*1.0/FLAGS.num_iter,
                        #'eps_iter': 3.*eps_/FLAGS.num_iter,
                        'eps_iter': 2. / 256.,
                       'clip_min': clip_min,
                       'clip_max': clip_max,
                       'nb_iter': FLAGS.num_iter,
                       'y_target': y_target}
        adv_x = tf.stop_gradient(att.generate(x_place, **att_params))
        preds = model(adv_x)
        if FLAGS.target==False:
            acc = model_eval(sess, x_place, y_place, preds, x_test, y_test, args=eval_par)
            print('adv_acc: %.3f' %acc)
        else:
            acc = model_eval_targetacc(sess, x_place, y_place, y_target, preds, x_test, y_test, y_test_target, args=eval_par)
            print('adv_acc_target: %.3f' %acc)
