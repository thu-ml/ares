from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):
    ''' An abstract interface for classifier model. '''

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
        ''' Initialize the abstract Classifier with metadata.

        :param n_class: A ``int`` number. Number of class of the classifier.
        :param x_min: A ``float`` number. Min value for the classifier's input.
        :param x_max: A ``float`` number. Max value for the classifier's input.
        :param x_shape: A ``tuple`` of ``int`` numbers. The shape of the classifier's input.
        :param x_dtype: A ``tf.DType`` instance. The data type of the classifier's input.
        :param y_dtype: A ``tf.DType`` instance. The data type of the classifier's classification result.
        '''
        self.n_class = n_class
        self.x_min, self.x_max, self.x_shape, self.x_dtype = x_min, x_max, x_shape, x_dtype
        self.y_dtype = y_dtype
        # cache labels output to reuse graph
        self._xs_labels_map = dict()

    @abstractmethod
    def _labels(self, xs):
        ''' Take an input ``tf.Tensor`` and give the classifier's classification result as a ``tf.Tensor``.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A ``tf.Tensor`` instance with shape of ``(self.n_class,)`` and with data type of ``self.y_dtype``.
            Represents the classification result.
        '''

    def labels(self, xs):
        ''' A wrapper for ``self._labels()`` to reuse graph.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A ``tf.Tensor`` instance with shape of ``(self.n_class,)`` and with data type of ``self.y_dtype``.
            Represents the classification result.
        '''
        try:
            return self._xs_labels_map[xs]
        except KeyError:
            labels = self._labels(xs)
            self._xs_labels_map[xs] = labels
            return labels


class ClassifierWithLogits(Classifier, metaclass=ABCMeta):
    ''' An abstract interface for classifier model which provides (maybe differentiable) logits output. '''

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
        ''' Initialize the abstract ClassifierWithLogits with metadata.

        :param n_class: A ``int`` number. Number of class of the classifier.
        :param x_min: A ``float`` number. Min value for the classifier's input.
        :param x_max: A ``float`` number. Max value for the classifier's input.
        :param x_shape: A ``tuple`` of ``int`` numbers. The shape of the classifier's input.
        :param x_dtype: A ``tf.DType`` instance. The data type of the classifier's input.
        :param y_dtype: A ``tf.DType`` instance. The data type of the classifier's classification result.
        '''
        super().__init__(n_class, x_min, x_max, x_shape, x_dtype, y_dtype)
        # cache logits and labels output to reuse graph
        self._xs_logits_labels_map = dict()

    @abstractmethod
    def _logits_and_labels(self, xs):
        ''' Take an input ``tf.Tensor`` and give the classifier's logits output as a ``tf.Tensor`` and classification
        result as a ``tf.Tensor``.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A tuple of two tensor, which represent the logits and the labels output of the classifier.
        '''

    def _labels(self, xs):
        ''' Implementation for the ``Classifier`` interface. '''
        _, labels = self._logits_and_labels(xs)
        return labels

    def logits_and_labels(self, xs):
        ''' A wrapper for ``self._logits_and_labels()`` to reuse graph.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A tuple of two tensor, which represent the logits and the labels output of the classifier.
        '''
        try:
            logits, labels = self._xs_logits_labels_map[xs]
        except KeyError:
            logits, labels = self._logits_and_labels(xs)
            self._xs_logits_labels_map[xs] = (logits, labels)
        return logits, labels

    def labels(self, xs):
        ''' A wrapper for ``self._logits_and_labels()`` to reuse graph which returns only labels output.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A tensor, which represent the labels output of the classifier.
        '''
        _, labels = self.logits_and_labels(xs)
        return labels

    def logits(self, xs):
        ''' A wrapper for ``self._logits_and_labels()`` to reuse graph which returns only logits output.

        :param xs: A ``tf.Tensor`` instance. Input of the classifier with shape of ``self.x_shape`` and with data type
            of ``self.x_dtype``. ``xs`` shall be in the range of [``self.x_min``, ``self.x_max``].
        :return: A tensor, which represent the logits output of the classifier.
        '''
        logits, _ = self.logits_and_labels(xs)
        return logits
