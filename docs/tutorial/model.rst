Defining & Using Models
=======================

Interface for Model
-------------------

Please read documentation for :doc:`../api/ares.model`. All models under ``example/cifar10/`` and ``example/imagenet/`` might be helpful, too.

The :class:`Classifier <ares.model.base.Classifier>` defines an abstract base class for an image classifier output only labels:

.. code-block:: python

   class Classifier(metaclass=ABCMeta):
       def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
           ...

       @abstractmethod
       def _labels(self, xs):
           ...

The :class:`ClassifierWithLogits <ares.model.base.ClassifierWithLogits` defines an abstract base class for an image classifier output not only labels, but also logits:

.. code-block:: python

   class ClassifierWithLogits(Classifier, metaclass=ABCMeta):
       def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
           ...

       @abstractmethod
       def _logits_and_labels(self, xs):
           ...

Defining Model in TensorFlow
----------------------------

Here is a brief example from ``example/cifar10/resnet56.py``:

.. code-block:: python

   class ResNet56(ClassifierWithLogits):
       def __init__(self):
           ClassifierWithLogits.__init__(self,
                                         x_min=0.0, x_max=1.0, x_shape=(32, 32, 3,), x_dtype=tf.float32,
                                         y_dtype=tf.int32, n_class=10)
   
       def _logits_and_labels(self, xs_ph):
           logits = ...  # calculate the logits of input xs_ph
           predicted_labels = tf.argmax(logits, 1, output_type=tf.int32)  # y_dtype is tf.int32
           return logits, predicted_labels

You need to pass meta information about the new model (data types, shape of input, numerical range of input) to the abstract base class in ``__init__()`` method. You need to return logits and labels for input ``xs_ph`` in ``_logits_and_labels()`` method. That's all.

For :class:`Classifier <ares.model.base.Classifier>`, the only difference is, only labels are needed in ``_labels()`` method.

Defining Model in PyTorch
-------------------------

We provide a decorator :meth:`pytorch_classifier_with_logits <ares.model.pytorch_wrapper.pytorch_classifier_with_logits>` which wraps a ``torch.nn.Module`` into a :class:`Classifier <ares.model.base.Classifier>` with differentiable logits output.

Here is a brief example from ``example/cifar10/wideresnet_trades.py``:

.. code-block:: python

   @pytorch_classifier_with_logits(n_class=10, x_min=0.0, x_max=1.0,
                                   x_shape=(32, 32, 3), x_dtype=tf.float32, y_dtype=tf.int32)
   class WideResNet_TRADES(torch.nn.Module):

       def __init__(self):
           ...
   
       def forward(self, x):
           ...

If you already have a PyTorch model defined as ``torch.nn.Module``, apply the decorator with meta information as its parameters to the PyTorch model.

.. note::

   In PyTorch, images are represented as ``[channels, height, width]``. In TensorFlow, images are represented as ``[height, width, channels]``. We use the TensorFlow way, so the model needs to handle the convertion.

Using Models
------------

Always use the ``logits()``, ``labels()`` and ``logits_and_labels()`` methods instead of the ``_labels()`` and ``_logits_and_labels()`` methods you defined. These methods without ``_`` prefix would cache results and avoid recalculating logits and labels for same input tensor.

Besides loading models manually, we provide :meth:`load_model_from_path <ares.model.loader.load_model_from_path>` to aid loading a model from a Python file. A global function ``load(session)`` should be defined inside the python file, which loads the model into the ``session`` and returns the model instance. Here is an example from ``example/cifar10/resnet56.py``:

.. code-block:: python

   def load(session):
       model = ResNet56()
       model.load(MODEL_PATH, session)
       return model

This way of loading models are used by ares's command line interface. See :doc:`benchmark` for more information.

All models under ``example/`` could be loaded using :meth:`load_model_from_path <ares.model.loader.load_model_from_path>`. For example:

.. code-block:: python

   rs_model = load_model_from_path('example/cifar10/resnet56.py')
   model = rs_model.load(session)

.. note::

   This function is kind of a dirty hack. It tries to handle relative import inside the Python file correctly. When possible, please avoid relative import, especially when there are name conflicts. But if you have to, please check ``example/imagenet/inception_v3.py`` for Python ``PATH`` hacks.

