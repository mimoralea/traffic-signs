from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import pickle

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import classifier

FLAGS = tf.app.flags.FLAGS

def eval_once(saver, summary_writer, top_k_op,
              summary_op, X_test, y_test, images_p, labels_p, keep_prob):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        # Restores from checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        predictions = sess.run(top_k_op, feed_dict={images_p:X_test, labels_p:y_test, keep_prob:1.0})
        print(predictions)
        true_count = np.sum(predictions)
        precision = true_count / len(X_test)
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op, feed_dict={images_p:X_test, labels_p:y_test, keep_prob:1.0}))
        summary.value.add(tag='Test Precision', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

    return precision

def eval_img(saver, logits, img, label, images_p, labels_p, keep_prob):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        # Restores from checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        sign_strings = pd.read_csv('signnames.csv', index_col=0)
        probs = sess.run(logits, feed_dict={images_p:[img], labels_p:[label], keep_prob:1.0})[0]
        sign_strings['prob'] = pd.Series(probs, index=sign_strings.index)
        sign_strings['correct'] = pd.Series(sign_strings.index.values == label, index=sign_strings.index)
        return sign_strings.sort_values(['prob', 'SignName'], ascending=[0, 1])

def evaluate(X_test, y_test):
    
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:

        image_shape = X_test[0].shape

        img_width = image_shape[0]
        img_height = image_shape[1]
        img_channels = image_shape[2]

        # Get images and labels for CIFAR-10.
        images_p = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_channels])
        images = images_p
        images = tf.map_fn(lambda img: tf.squeeze(tf.image.rgb_to_grayscale(img)), images)
        images = tf.expand_dims(images, -1)
        labels_p = tf.placeholder(tf.int32, shape=[None])
        labels = labels_p
        keep_prob = tf.placeholder(tf.float32)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = classifier.inference(images, keep_prob)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 5)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            classifier.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        #precision = eval_once(saver, summary_writer, top_k_op, summary_op,
        #                      X_test, y_test, images, labels)
        #print('%s: test set precision %.3f' % (datetime.now(), precision))

        idx = 123
        img = X_test[idx]
        label = y_test[idx]
        predictions = eval_img(saver, logits,
                               img, label,
                               images_p, labels_p, keep_prob)
        print(predictions)
        plt.style.use('ggplot')

        plt.imshow(img)
        plt.show()


def main(argv=None):  # pylint: disable=unused-argument

    training_file = 'data/train.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    image_shape = X_train[0].shape
    n_classes = len(np.unique(y_train))

    img_width = image_shape[0]
    img_height = image_shape[1]
    img_channels = image_shape[2]

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    evaluate(X_test, y_test)


if __name__ == '__main__':
    tf.app.run()
