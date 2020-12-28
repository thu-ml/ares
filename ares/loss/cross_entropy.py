import tensorflow as tf

from ares.loss.base import Loss


class CrossEntropyLoss(Loss):
    ''' Cross entropy loss. '''

    def __init__(self, model):
        ''' Initialize CrossEntropyLoss.

        :param model: An instance of ``ClassifierWithLogits``.
        '''
        self.model = model

    def __call__(self, xs, ys):
        logits = self.model.logits(xs)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits)
        return loss


class EnsembleCrossEntropyLoss(Loss):
    ''' Ensemble multiple models' cross entropy loss. '''

    def __init__(self, models, weights):
        ''' Initialize EnsembleCrossEntropyLoss.

        :param models: A list of ``ClassifierWithLogits``.
        :param weights: Weights for ensemble these models.
        '''
        self.models, self.weights = models, weights

    def __call__(self, xs, ys):
        losses = []
        for model, weight in zip(self.models, self.weights):
            logits = model.logits(xs)
            losses.append(weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits))
        return tf.reduce_sum(losses, axis=0)


class EnsembleRandomnessCrossEntropyLoss(Loss):
    ''' Ensemble a random model's cross entropy loss. '''

    def __init__(self, model, n, session):
        ''' Initialize EnsembleRandomnessCrossEntropyLoss.

        :param model: An instance of ``ClassifierWithLogits``.
        :param n: Number of samples to ensemble.
        :param session: ``tf.Session``.
        '''
        assert(n > 1)

        self.model, self.n = model, n
        self._session = session

    def __call__(self, xs, ys):
        d_output_ph = tf.placeholder(dtype=xs.dtype)

        xs_ph = tf.placeholder(dtype=xs.dtype, shape=xs.shape)
        ys_ph = tf.placeholder(dtype=ys.dtype, shape=ys.shape)

        logits = self.model.logits(xs_ph)

        one_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys_ph, logits=logits)
        one_loss_grads = tf.gradients(one_loss, xs_ph, grad_ys=[d_output_ph])[0]

        @tf.custom_gradient
        def fn_loss(xs_tf, ys_tf):
            xs_np = xs_tf.numpy()
            ys_np = ys_tf.numpy()

            loss_np = self._session.run(one_loss, feed_dict={xs_ph: xs_np, ys_ph: ys_np})
            for _ in range(self.n - 1):
                loss_np += self._session.run(one_loss, feed_dict={xs_ph: xs_np, ys_ph: ys_np})
            loss_np /= self.n

            def fn_loss_grads(d_output_tf):
                d_output_np = d_output_tf.numpy()

                loss_grads_np = self._session.run(
                    one_loss_grads, feed_dict={xs_ph: xs_np, ys_ph: ys_np, d_output_ph: d_output_np})
                for _ in range(self.n - 1):
                    loss_grads_np += self._session.run(
                        one_loss_grads, feed_dict={xs_ph: xs_np, ys_ph: ys_np, d_output_ph: d_output_np})
                loss_grads_np /= float(self.n)

                # Here the '1' should be 'None', since there is actually no gradient for the second parameter ys, but
                # tensorflow converts 'None' to '0.0', and then tries converting '0.0' to ys' data type ('tf.int32'),
                # which gives an error. So we put a valid integer here to workaround this strange behavior.
                return tf.convert_to_tensor(loss_grads_np), 1

            return tf.convert_to_tensor(loss_np), fn_loss_grads

        loss = tf.py_function(func=fn_loss, inp=[xs, ys], Tout=one_loss.dtype)
        loss.set_shape(one_loss.shape)

        return loss
