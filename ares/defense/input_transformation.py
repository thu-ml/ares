''' A general wrapper for input transformation based defense. '''

from ares.model.base import Classifier, ClassifierWithLogits


def input_transformation(rs_class, transform, args_fn, kwargs_fn):
    ''' Apply input transformation to ``rs_class`` to get a new classifier.

    :param rs_class: the classifier class to apply the input transformation, which should be subclass of Classifier.
        When the logits is available, the returned new classifier would implement the ClassifierWithLogits interface.
    :param transform: The transformation to apply to the classifier's input. It should be a function, whose first
        parameter is the input tensor (in batch) for the classifier and returns the transformed input tensor. Extra
        parameters returned by ``args_fn`` and ``kwargs_fn`` are passed to this function following the input tensor.
    :param args_fn: A function returns extra parameters for the ``transform`` function, whose parameter is the
        classifier instance.
    :param kwargs_fn: A function returns extra keyword parameters for the ``transform`` function, whose parameter is the
        classifier instance.
    :return: A new classifier with the input transformation applied.
    '''
    if issubclass(rs_class, ClassifierWithLogits):  # prefer using ClassifierWithLogits
        class Wrapper(rs_class):  # directly inherit the classifier's class
            def _logits_and_labels(self, xs):  # implement ClassifierWithLogits' interface
                args, kwargs = args_fn(self), kwargs_fn(self)
                xs_transformed = transform(xs, *args, **kwargs)
                # we need to call the _logits_and_labels() instead of logits_and_labels() here
                return super()._logits_and_labels(xs_transformed)
        return Wrapper
    elif issubclass(rs_class, Classifier):
        class Wrapper(rs_class):  # directly inherit the classifier's class
            def _labels(self, xs):  # implement Classifier's interface
                args, kwargs = args_fn(self), kwargs_fn(self)
                xs_transformed = transform(xs, *args, **kwargs)
                # we need to call the _logits() instead of logits() here
                return super()._labels(xs_transformed)
        return Wrapper
    else:
        raise TypeError('input_transformation() requires a Classifier or a ClassifierWithLogits class.')
