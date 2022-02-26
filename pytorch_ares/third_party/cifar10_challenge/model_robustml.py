import robustml
import tensorflow as tf

import model

class Model(robustml.model.Model):
  def __init__(self, sess):
    self._model = model.Model('eval')

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('models/model_0')
    saver.restore(sess, checkpoint)

    self._sess = sess
    self._input = self._model.x_input
    self._logits = self._model.pre_softmax
    self._predictions = self._model.predictions
    self._dataset = robustml.dataset.CIFAR10()
    self._threat_model = robustml.threat_model.Linf(epsilon=0.03)

  @property
  def dataset(self):
      return self._dataset

  @property
  def threat_model(self):
      return self._threat_model

  def classify(self, x):
      return self._sess.run(self._predictions,
                            {self._input: x})[0]

  # expose attack interface

  @property
  def input(self):
      return self._input

  @property
  def logits(self):
      return self._logits

  @property
  def predictions(self):
      return self._predictions
