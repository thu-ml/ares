"""
Model construction utilities based on keras
"""
import warnings
from distutils.version import LooseVersion
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from cleverhans.model import Model, NoSuchLayerError
import tensorflow as tf


class KerasModelWrapper(Model):
    """
  An implementation of `Model` that wraps a Keras model. It
  specifically exposes the hidden features of a model by creating new models.
  The symbolic graph is reused and so there is little overhead. Splitting
  in-place operations can incur an overhead.
  """

    def __init__(self, model, num_class=10):
        """
        Create a wrapper for a Keras model
        :param model: A Keras model
        """
        super(KerasModelWrapper, self).__init__()

        if model is None:
            raise ValueError('model argument must be supplied.')

        self.model = model
        self.keras_model = None
        self.num_classes = num_class

    def _get_softmax_name(self):
        """
    Looks for the name of the softmax layer.
    :return: Softmax layer name
    """
        for layer in self.model.layers:
            cfg = layer.get_config()
            if cfg['name'] == 'average_1':
                return layer.name

        raise Exception("No softmax layers found")

    def _get_logits_name(self):
        """
    Looks for the name of the layer producing the logits.
    :return: name of layer producing the logits
    """
        softmax_name = self._get_softmax_name()
        softmax_layer = self.model.get_layer(softmax_name)

        if not isinstance(softmax_layer, Activation):
            # In this case, the activation is part of another layer
            return softmax_name

        if hasattr(softmax_layer, 'inbound_nodes'):
            warnings.warn(
                "Please update your version to keras >= 2.1.3; "
                "support for earlier keras versions will be dropped on "
                "2018-07-22")
            node = softmax_layer.inbound_nodes[0]
        else:
            node = softmax_layer._inbound_nodes[0]

        logits_name = node.inbound_layers[0].name

        return logits_name

    def get_logits(self, x):
        """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the logits
    """
        # logits_name = self._get_logits_name()
        # logits_layer = self.get_layer(x, logits_name)

        # # Need to deal with the case where softmax is part of the
        # # logits layer
        # if logits_name == self._get_softmax_name():
        #   softmax_logit_layer = self.get_layer(x, logits_name)

        #   # The final op is the softmax. Return its input
        #   logits_layer = softmax_logit_layer._op.inputs[0]
        prob = self.get_probs(x)
        logits = tf.log(prob)
        #logits = prob
        return logits

    def get_probs(self, x):
        """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the probs
    """

        return self.model(x)

    def get_layer_names(self):
        """
    :return: Names of all the layers kept by Keras
    """
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def fprop(self, x):
        """
    Exposes all the layers of the model returned by get_layer_names.
    :param x: A symbolic representation of the network input
    :return: A dictionary mapping layer names to the symbolic
             representation of their output.
    """
        from keras.models import Model as KerasModel

        if self.keras_model is None:
            # Get the input layer
            new_input = self.model.get_input_at(0)

            # Make a new model that returns each of the layers as output
            out_layers = [x_layer.output for x_layer in self.model.layers]
            self.keras_model = KerasModel(new_input, out_layers)

        # and get the outputs for that model on the input x
        outputs = self.keras_model(x)

        # Keras only returns a list for outputs of length >= 1, if the model
        # is only one layer, wrap a list
        if len(self.model.layers) == 1:
            outputs = [outputs]

        # compute the dict to return
        fprop_dict = dict(zip(self.get_layer_names(), outputs))

        return fprop_dict

    def get_layer(self, x, layer):
        """
    Expose the hidden features of a model given a layer name.
    :param x: A symbolic representation of the network input
    :param layer: The name of the hidden layer to return features at.
    :return: A symbolic representation of the hidden features
    :raise: NoSuchLayerError if `layer` is not in the model.
    """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested