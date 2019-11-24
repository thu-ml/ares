from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):
    """
    An abstract interface for classifier model.
    """

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
        """
        :param n_class: A `int` number. Number of class of the classifier.
        :param x_min: A `float` number. Min value for the classifier's input.
        :param x_max: A `float` number. Max value for the classifier's input.
        :param x_shape: A `tuple` of `int` numbers. The shape of the classifier's input.
        :param x_dtype: A `tf.DType` instance. The data type of the classifier's input.
        :param y_dtype: A `tf.DType` instance. The data type of the classifier's classification result.
        """
        self.n_class = n_class
        self.x_min, self.x_max, self.x_shape, self.x_dtype = x_min, x_max, x_shape, x_dtype
        self.y_dtype = y_dtype
        self._xs_lbs_map = dict()

    @abstractmethod
    def _labels(self, xs):
        """
        Take an input `tf.Tensor` and give the classifier's classification result as a `tf.Tensor`.
        :param xs: A `tf.Tensor` instance. Input of the classifier with shape of `self.x_shape` and with data type of
            `self.x_dtype`. `xs` shall be in the range of [`self.x_min`, `self.x_max`].
        :return: A `tf.Tensor` instance with shape of `(self.n_class,)` and with data type of `self.y_dtype`. Represents
            the classification result.
        """
        pass

    def labels(self, xs):
        """
        A wrapper for `self._labels()` to reuse graph.
        :param xs: A `tf.Tensor` instance. Input of the classifier with shape of `self.x_shape` and with data type of
            `self.x_dtype`. `xs` shall be in the range of [`self.x_min`, `self.x_max`].
        :return: A `tf.Tensor` instance with shape of `(self.n_class,)` and with data type of `self.y_dtype`. Represents
            the classification result.
        """
        try:
            return self._xs_lbs_map[xs]
        except KeyError:
            lbs = self._labels(xs)
            self._xs_lbs_map[xs] = lbs
            return lbs


class ClassifierWithLogits(Classifier, metaclass=ABCMeta):
    """
    An abstract interface for classifier model which provides (maybe differentiable) logits output.
    """

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype):
        """
        :param n_class: A `int` number. Number of class of the classifier.
        :param x_min: A `float` number. Min value for the classifier's input.
        :param x_max: A `float` number. Max value for the classifier's input.
        :param x_shape: A `tuple` of `int` numbers. The shape of the classifier's input.
        :param x_dtype: A `tf.DType` instance. The data type of the classifier's input.
        :param y_dtype: A `tf.DType` instance. The data type of the classifier's classification result.
        """
        super().__init__(n_class, x_min, x_max, x_shape, x_dtype, y_dtype)
        self._xs_lgs_lbs_map = dict()

    @abstractmethod
    def _logits_and_labels(self, xs):
        """
        Take an input `tf.Tensor` and give the classifier's logits output as a `tf.Tensor` and classification result as
        a `tf.Tensor`.
        :param xs: A `tf.Tensor` instance. Input of the classifier with shape of `self.x_shape` and with data type of
            `self.x_dtype`. `xs` shall be in the range of [`self.x_min`, `self.x_max`].
        :return: TODO
        """
        pass

    def _labels(self, xs):
        _, lbs = self._logits_and_labels(xs)
        return lbs

    def logits_and_labels(self, xs):
        """
        A wrapper for `self._logits_and_labels()` to reuse graph.
        :param xs: A `tf.Tensor` instance. Input of the classifier with shape of `self.x_shape` and with data type of
            `self.x_dtype`. `xs` shall be in the range of [`self.x_min`, `self.x_max`].
        :return: TODO
        """
        try:
            lgs, lbs = self._xs_lgs_lbs_map[xs]
        except KeyError:
            lgs, lbs = self._logits_and_labels(xs)
            self._xs_lgs_lbs_map[xs] = (lgs, lbs)
        return lgs, lbs

    def labels(self, xs):
        _, lbs = self.logits_and_labels(xs)
        return lbs
