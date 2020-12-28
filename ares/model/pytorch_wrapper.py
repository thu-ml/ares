import tensorflow as tf
import torch

from ares.model.base import ClassifierWithLogits


def pytorch_classifier_with_logits(n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
    ''' A decorator for wrapping a pytorch model class into a ``ClassifierWithLogits``.

    The parameters provide metadata for the classifier, which are passed to the ``ClassifierWithLogits`` interface. The
    decorated pytorch class should be an instance of ``torch.nn.Module``, which provides both forward and backward
    capacity. However, due to some limitation (or bug) of ``@tf.custom_gradient``, we cannot backpropagate multiple
    gradients through the wrapped model in one ``session.run()``, even though pytorch support this feature via the
    ``retain_graph`` parameter.

    :param n_class: A ``int`` number. Number of class of the classifier.
    :param x_min: A ``float`` number. Min value for the classifier's input.
    :param x_max: A ``float`` number. Max value for the classifier's input.
    :param x_shape: A ``tuple`` of ``int`` numbers. The shape of the classifier's input.
    :param x_dtype: A ``tf.DType`` instance. The data type of the classifier's input.
    :param y_dtype: A ``tf.DType`` instance. The data type of the classifier's classification result.
    '''
    def decorator(nn_class):  # the inner decorator
        class Wrapper(ClassifierWithLogits):  # Wrapper class for the pytorch model class `nn_class`
            def __init__(self, *args, **kwargs):  # we need to pass all parameters down to the pytorch model
                super().__init__(n_class, x_min, x_max, x_shape, x_dtype, y_dtype)
                # create the inner model
                self._inner = nn_class(*args, **kwargs)

                # Since we need the logits_torch calculated in the forward pass in the backward pass, eager mode is
                # required. Or we need to run the forward pass both in calculating the logits and calculating the logits
                # gradients with relate to the input.
                @tf.custom_gradient
                def eager_tf_logits(xs_tf):  # xs_tf is an eager tensor
                    xs_np = xs_tf.numpy()
                    xs_torch = torch.autograd.Variable(torch.from_numpy(xs_np), requires_grad=True)
                    logits_torch = self._inner(xs_torch)  # the forward pass

                    def eager_tf_logits_grad(d_output_tf):   # d_output_tf is an eager tensor
                        # We do NOT use retain_graph=True here, since the @tf.custom_gradient for eager function does
                        # NOT support running multiple backward pass. This (bug) brings some limitation on this wrapper:
                        # we cannot backpropagate multiple gradients through this model in one session.run().
                        logits_torch.backward(torch.from_numpy(d_output_tf.numpy()))  # the backward pass
                        return tf.convert_to_tensor(xs_torch.grad.data.detach().numpy())

                    return tf.convert_to_tensor(logits_torch.detach().numpy()), eager_tf_logits_grad

                self._eager_tf_logits = eager_tf_logits

            def _logits_and_labels(self, xs):  # implement ClassifierWithLogits' interface
                # wrap the eager mode function into a normal tensorflow op with tf.py_function
                logits = tf.py_function(func=self._eager_tf_logits, inp=[xs], Tout=self.x_dtype)
                logits.set_shape((xs.shape[0], self.n_class))
                labels = tf.argmax(logits, 1, output_type=self.y_dtype)
                return logits, labels

            def __getattr__(self, name):  # passing down method calls to inner pytorch model
                return getattr(self._inner, name)

        return Wrapper

    return decorator
