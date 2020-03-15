from realsafe.dataset import imagenet
import tensorflow as tf

dataset = imagenet.load_dataset(299, 299)
iterator = dataset.batch(10).make_one_shot_iterator().get_next()

session = tf.Session()

while True:
    filenames, xs, ys = session.run(iterator)
    print(filenames)
