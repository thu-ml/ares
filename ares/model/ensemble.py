''' Provide helper classes for ensemble (random) model(s). '''

import numpy as np
import tensorflow as tf

from ares.model.base import Classifier


class EnsembleModel(Classifier):
    ''' Ensemble multiple ``ClassifierWithLogits`` by averaging theirs output probabilities for each label. '''

    def __init__(self, models, weights):
        ''' Initialize EnsembleModel.

        :param models: A list of ``ClassifierWithLogits`` to ensemble.
        :param weights: Weight for averaging these models' outputs.
        '''
        super().__init__(models[0].n_class,
                         models[0].x_min, models[0].x_max, models[0].x_shape, models[0].x_dtype, models[0].y_dtype)
        self.models, self.weights = models, weights

    def _labels(self, xs):
        ps = [weight * tf.nn.softmax(model.logits(xs)) for model, weight in zip(self.models, self.weights)]
        return tf.cast(tf.argmax(tf.reduce_sum(ps, axis=0), 1), self.models[0].y_dtype)


class EnsembleRandomnessModel(Classifier):
    ''' Ensemble a random ``ClassifierWithLogits`` by averaging its output probabilities for each label. '''

    def __init__(self, model, n, session):
        ''' Initialize EnsembleRandomnessModel.

        :param model: A ``ClassifierWithLogits`` to ensemble.
        :param n: Number of samples per input.
        :param session: ``tf.Session``.
        '''
        super().__init__(model.n_class, model.x_min, model.x_max, model.x_shape, model.x_dtype, model.y_dtype)
        self.model = model
        self.n = n
        self._session = session

    def _labels(self, xs):
        xs_ph = tf.placeholder(xs.dtype, shape=xs.shape)
        one_ps = tf.nn.softmax(self.model.logits(xs_ph))

        def fn_labels(xs_tf):
            xs_np = xs_tf.numpy()
            ps = self._session.run(one_ps, feed_dict={xs_ph: xs_np})
            for _ in range(self.n - 1):
                ps += self._session.run(one_ps, feed_dict={xs_ph: xs_np})
            return tf.convert_to_tensor(np.argmax(ps, 1), dtype=self.model.y_dtype)

        labels = tf.py_function(fn_labels, [xs], self.model.y_dtype)
        labels.set_shape(xs.shape[0])

        return labels
