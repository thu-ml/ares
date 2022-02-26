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

FLAGS = tf.app.flags.FLAGS

#Common Flags for two models
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch_size for attack')
tf.app.flags.DEFINE_string('optimizer', 'mom', '')
tf.app.flags.DEFINE_string('attack_method', 'FastGradientMethod', '')
tf.app.flags.DEFINE_integer('version', 2, '')
tf.app.flags.DEFINE_float('lr', 0.01, 'initial lr')
tf.app.flags.DEFINE_bool('target', True, 'is target attack or not')
tf.app.flags.DEFINE_integer('num_iter', 10, '')
tf.app.flags.DEFINE_string('dataset', 'cifar10', '')
tf.app.flags.DEFINE_bool('use_random', False, 'whether use random center or MMLDA center in the network')
tf.app.flags.DEFINE_bool('use_dense', True, 'whether use extra dense layer in the network')
tf.app.flags.DEFINE_bool('use_leaky', False, 'whether use leaky relu in the network')
tf.app.flags.DEFINE_integer('epoch', 180, 'the epoch of model to load')
tf.app.flags.DEFINE_bool('use_BN', True, 'whether use batch normalization in the network')

# SCE, MMC-10, MMC-100, AT-SCE, AT-MMC-10, AT-MMC-100
tf.app.flags.DEFINE_string('model_1', 'SCE', '')
tf.app.flags.DEFINE_string('model_2', 'MMC-10', '')

#Specific Flags for model 1
tf.app.flags.DEFINE_float('mean_var_1', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_string('attack_method_for_advtrain_1', 'FastGradientMethod', '')
tf.app.flags.DEFINE_bool('use_target_1', False, 'whether use target attack or untarget attack for adversarial training')
tf.app.flags.DEFINE_bool('use_ball_1', True, 'whether use ball loss or softmax')
tf.app.flags.DEFINE_bool('use_MMLDA_1', True, 'whether use MMLDA or softmax')
tf.app.flags.DEFINE_bool('use_advtrain_1', True, 'whether use advtraining or normal training')
tf.app.flags.DEFINE_float('adv_ratio_1', 1.0, 'the ratio of adversarial examples in each mini-batch')
tf.app.flags.DEFINE_bool('normalize_output_for_ball_1', True, 'whether apply softmax in the inference phase')

#Specific Flags for model 2
tf.app.flags.DEFINE_float('mean_var_2', 10, 'parameter in MMLDA')
tf.app.flags.DEFINE_string('attack_method_for_advtrain_2', 'FastGradientMethod', '')
tf.app.flags.DEFINE_bool('use_target_2', False, 'whether use target attack or untarget attack for adversarial training')
tf.app.flags.DEFINE_bool('use_ball_2', True, 'whether use ball loss or softmax')
tf.app.flags.DEFINE_bool('use_MMLDA_2', True, 'whether use MMLDA or softmax')
tf.app.flags.DEFINE_bool('use_advtrain_2', True, 'whether use advtraining or normal training')
tf.app.flags.DEFINE_float('adv_ratio_2', 1.0, 'the ratio of adversarial examples in each mini-batch')
tf.app.flags.DEFINE_bool('normalize_output_for_ball_2', True, 'whether apply softmax in the inference phase')

##### model 1 is the substitute model used to craft adversarial examples, model 2 is the original model used to classify these adversarial examples.

def return_paras(model_name):
    if model_name == 'SCE':
        return 0, None, False, False, False, False, 0.0, True
    elif model_name == 'MMC-10':
        return 10.0, None, False, True, True, False, 0.0, False
    elif model_name == 'MMC-100':
        return 100.0, None, False, True, True, False, 0.0, False
    elif model_name == 'AT-SCE':
        return 0, 'MadryEtAl', True, False, False, True, 1.0, True
    elif model_name == 'AT-MMC-10':
        return 10, 'MadryEtAl', True, True, True, True, 1.0, False
    elif model_name == 'AT-MMC-100':
        return 100, 'MadryEtAl', True, True, True, True, 1.0, False
    else:
        return None


FLAGS.mean_var_1, FLAGS.attack_method_for_advtrain_1, FLAGS.use_target_1, FLAGS.use_ball_1, \
FLAGS.use_MMLDA_1, FLAGS.use_advtrain_1, FLAGS.adv_ratio_1, FLAGS.normalize_output_for_ball_1 = return_paras(FLAGS.model_1)


FLAGS.mean_var_2, FLAGS.attack_method_for_advtrain_2, FLAGS.use_target_2, FLAGS.use_ball_2, \
FLAGS.use_MMLDA_2, FLAGS.use_advtrain_2, FLAGS.adv_ratio_2, FLAGS.normalize_output_for_ball_2 = return_paras(FLAGS.model_2)



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


if FLAGS.use_BN==True:
    BN_name = '_withBN'
    print('Use BN in the model')
else:
    BN_name = '_noBN'
    print('Do not use BN in the model')

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
mean_logits_1 = FLAGS.mean_var_1 * tf.constant(mean_logits,dtype=tf.float32)
mean_logits_2 = FLAGS.mean_var_2 * tf.constant(mean_logits,dtype=tf.float32)


#MMLDA prediction function
def MMLDA_layer_1(x, means=mean_logits_1, num_class=num_class, use_ball=FLAGS.use_ball_1):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    if use_ball==True:
        if FLAGS.normalize_output_for_ball_1==False:
            return logits
        else:
            return tf.nn.softmax(logits, axis=-1)
    else:
        return tf.nn.softmax(logits, axis=-1)


def MMLDA_layer_2(x, means=mean_logits_2, num_class=num_class, use_ball=FLAGS.use_ball_2):
    #x_shape = batch_size X num_dense
    x_expand = tf.tile(tf.expand_dims(x,axis=1),[1,num_class,1]) #batch_size X num_class X num_dense
    mean_expand = tf.expand_dims(means,axis=0) #1 X num_class X num_dense
    logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1) #batch_size X num_class
    if use_ball==True:
        if FLAGS.normalize_output_for_ball_2==False:
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
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)
y_test_target = keras.utils.to_categorical(y_test_target, num_class)


# Define input TF placeholder
y_place = tf.placeholder(tf.float32, shape=(None, num_class))
y_target = tf.placeholder(tf.float32, shape=(None, num_class))
sess = tf.Session()
keras.backend.set_session(sess)


model_input_1 = Input(shape=input_shape)
model_input_2 = Input(shape=input_shape)

#dim of logtis is batchsize x dim_means
if version == 2:
    original_model_1,_,_,_,final_features_1 = resnet_v2(input=model_input_1, depth=depth, num_classes=num_class, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)
else:
    original_model_1,_,_,_,final_features_1 = resnet_v1(input=model_input_1, depth=depth, num_classes=num_class, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)

if version == 2:
    original_model_2,_,_,_,final_features_2 = resnet_v2(input=model_input_2, depth=depth, num_classes=num_class, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)
else:
    original_model_2,_,_,_,final_features_2 = resnet_v1(input=model_input_2, depth=depth, num_classes=num_class, \
                                                    use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense, use_leaky=FLAGS.use_leaky)





##### Load model 1 #####
#Whether use target attack for adversarial training
if FLAGS.use_target_1==False:
    is_target_1 = ''
else:
    is_target_1 = 'target'

if FLAGS.use_advtrain_1==True:
    dirr_1 = 'advtrained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain_1 = '_'+is_target_1+FLAGS.attack_method_for_advtrain_1
    adv_ratio_name_1 = '_advratio'+str(FLAGS.adv_ratio_1)
    mean_var_1 = int(FLAGS.mean_var_1)
else:
    dirr_1 = 'trained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain_1 = ''
    adv_ratio_name_1 = ''
    mean_var_1 = FLAGS.mean_var_1

if FLAGS.use_MMLDA_1==True:
    print('Using MMLDA for model 1, the substitute model')
    new_layer_1 = Lambda(MMLDA_layer_1)
    predictions_1 = new_layer_1(final_features_1)
    model_1 = Model(input=model_input_1, output=predictions_1)
    use_ball_1=''
    if FLAGS.use_ball_1==False:
        print('Using softmax function for model 1')
        use_ball_1='_softmax'
    filepath_dir_1 = dirr_1+'resnet32v'+str(version)+'_meanvar'+str(mean_var_1) \
                                                                +'_'+FLAGS.optimizer \
                                                                +'_lr'+str(FLAGS.lr) \
                                                                +'_batchsize'+str(FLAGS.batch_size) \
                                                                +attack_method_for_advtrain_1+adv_ratio_name_1+BN_name+name_leaky+name_dense+name_random+use_ball_1+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'
else:
    print('Using softmax loss for model 1')
    model_1 = original_model_1
    filepath_dir_1 = dirr_1+'resnet32v'+str(version)+'_'+FLAGS.optimizer \
                                                            +'_lr'+str(FLAGS.lr) \
                                                            +'_batchsize'+str(FLAGS.batch_size)+attack_method_for_advtrain_1+adv_ratio_name_1+BN_name+name_leaky+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'
wrap_ensemble_1 = KerasModelWrapper(model_1, num_class=num_class)
model_1.load_weights(filepath_dir_1)




##### Load model 2 #####
#Whether use target attack for adversarial training
if FLAGS.use_target_2==False:
    is_target_2 = ''
else:
    is_target_2 = 'target'

if FLAGS.use_advtrain_2==True:
    dirr_2 = 'advtrained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain_2 = '_'+is_target_2+FLAGS.attack_method_for_advtrain_2
    adv_ratio_name_2 = '_advratio'+str(FLAGS.adv_ratio_2)
    mean_var_2 = int(FLAGS.mean_var_2)
else:
    dirr_2 = 'trained_models/'+FLAGS.dataset+'/'
    attack_method_for_advtrain_2 = ''
    adv_ratio_name_2 = ''
    mean_var_2 = FLAGS.mean_var_2

if FLAGS.use_MMLDA_2==True:
    print('Using MMLDA for model 2, the original model')
    new_layer_2 = Lambda(MMLDA_layer_2)
    predictions_2 = new_layer_2(final_features_2)
    model_2 = Model(input=model_input_2, output=predictions_2)
    use_ball_2=''
    if FLAGS.use_ball_2==False:
        print('Using softmax function for model 2')
        use_ball_2='_softmax'
    filepath_dir_2 = dirr_2+'resnet32v'+str(version)+'_meanvar'+str(mean_var_2) \
                                                                +'_'+FLAGS.optimizer \
                                                                +'_lr'+str(FLAGS.lr) \
                                                                +'_batchsize'+str(FLAGS.batch_size) \
                                                                +attack_method_for_advtrain_2+adv_ratio_name_2+BN_name+name_leaky+name_dense+name_random+use_ball_2+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'
else:
    print('Using softmax loss for model 2')
    model_2 = original_model_2
    filepath_dir_2 = dirr_2+'resnet32v'+str(version)+'_'+FLAGS.optimizer \
                                                            +'_lr'+str(FLAGS.lr) \
                                                            +'_batchsize'+str(FLAGS.batch_size)+attack_method_for_advtrain_2+adv_ratio_name_2+BN_name+name_leaky+'/' \
                                                                +'model.'+str(FLAGS.epoch).zfill(3)+'.h5'
wrap_ensemble_2 = KerasModelWrapper(model_2, num_class=num_class)
model_2.load_weights(filepath_dir_2)






# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap_ensemble_1)
elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.FastGradientMethod(wrap_ensemble_1)
elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att = attacks.MomentumIterativeMethod(wrap_ensemble_1)
elif FLAGS.attack_method == 'BasicIterativeMethod':
    att = attacks.BasicIterativeMethod(wrap_ensemble_1)


# Consider the attack to be constant
eval_par = {'batch_size': FLAGS.batch_size}

for eps in range(2):
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
    preds = model_2(adv_x)
    if FLAGS.target==False:
        acc = model_eval(sess, x_place, y_place, preds, x_test, y_test, args=eval_par)
        print('adv_acc of model 1 transfer to model 2 is: %.3f' %acc)
    else:
        acc = model_eval_targetacc(sess, x_place, y_place, y_target, preds, x_test, y_test, y_test_target, args=eval_par)
        print('adv_acc_target of model 1 transfer to model 2 is: %.3f' %acc)
