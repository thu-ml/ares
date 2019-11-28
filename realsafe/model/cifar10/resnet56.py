import os
import tensorflow as tf
import numpy as np

from six.moves import urllib
from realsafe.model.base import ClassifierWithLogits


class Resnet(object):
    """
    we define the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    """

    def __init__(self, mode='eval'):
        self.mode = mode

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        """
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        """

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables

    def output_layer(self, input_layer, num_labels):
        """
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
        :return: output layer Y = WX + B
        """
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                                     initializer=tf.initializers.variance_scaling(distribution="uniform"))
        fc_b = self.create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        return fc_h

    def batch_normalization_layer(self, x):
        """
        batch normalization
        """
        return tf.contrib.layers.batch_norm(
            inputs=x,
            decay=.9,
            center=True,
            scale=True,
            activation_fn=None,
            updates_collections=None,
            is_training=(self.mode == 'train'))

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        """
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        """
        filter_var = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(input_layer, filter_var, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer)

        output = tf.nn.relu(bn_layer)
        return output

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        """
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        """

        bn_layer = self.batch_normalization_layer(input_layer)
        relu_layer = tf.nn.relu(bn_layer)

        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

    def residual_block(self, input_layer, output_channel, first_block=False):
        """
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        """
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = self.create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def inference(self, input_tensor_batch, n, reuse):
        """
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param input_tensor_batch: 4D tensor
        :param n: num_residual_blocks
        :param reuse: To build train graph, reuse=False.
        :return: last layer in the network. Not softmax-ed
        """
        mean, std = tf.nn.moments(input_tensor_batch, [1, 2, 3], keep_dims=True)
        std += tf.constant(1e-5, dtype=np.float32)
        input_tensor_batch = (input_tensor_batch - mean) / tf.sqrt(std)
        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = self.conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = self.residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = self.residual_block(layers[-1], 16)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = self.residual_block(layers[-1], 32)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = self.residual_block(layers[-1], 64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc', reuse=reuse):
            bn_layer = self.batch_normalization_layer(layers[-1])
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [64]
            output = self.output_layer(global_pool, 10)
            layers.append(output)

        return layers[-1]


class ResNet56(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self,
                                      x_min=0.0,
                                      x_max=1.0,
                                      x_shape=(32, 32, 3,),
                                      x_dtype=tf.float32,
                                      y_dtype=tf.int32,
                                      n_class=10)

        self.num_residual_blocks = 9
        self.model = Resnet('eval')

    def _logits_and_labels(self, xs_ph):
        logits = self.model.inference(xs_ph, self.num_residual_blocks, reuse=tf.AUTO_REUSE)
        predicts = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(predicts, 1, output_type=tf.int32)

        return logits, predicted_labels

    def load(self, session, **kwargs):
        x_input = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        self.model.inference(x_input, self.num_residual_blocks, reuse=False)

        model_path = kwargs['model_path']

        if not os.path.exists(model_path):
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            urllib.request.urlretrieve(
                'http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/resnet56-cifar.ckpt',
                model_path)

        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(session, model_path)


if __name__ == '__main__':
    with tf.Session() as sess:
        model = ResNet56()
