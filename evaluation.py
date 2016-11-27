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

def eval_once(saver, top_k_op, X_test, y_test, images_p, labels_p, keep_prob_p):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        predictions = sess.run(top_k_op,
                               feed_dict={images_p:X_test,
                                          labels_p:y_test,
                                          keep_prob_p:1.0})
        print(predictions)
        true_count = np.sum(predictions)
        precision = true_count / len(X_test)

    return precision

def eval_img(saver, logits, img, label, images_p, labels_p, keep_prob):
    with tf.Session() as sess:
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

def evaluate_image(img, label):
    with tf.Graph().as_default() as g:

        image_shape = img.shape
        img_width = image_shape[0]
        img_height = image_shape[1]
        img_channels = image_shape[2]

        images_p = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_channels])
        labels_p = tf.placeholder(tf.int32, shape=[None])
        keep_prob_p = tf.placeholder(tf.float32)

        logits = classifier.inference(images_p, keep_prob_p)

        variable_averages = tf.train.ExponentialMovingAverage(
            classifier.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        predictions = eval_img(saver, logits, img, label,
                               images_p, labels_p, keep_prob_p)
        print(predictions)
        plt.style.use('ggplot')

        plt.imshow(img)
        plt.show()


def evaluate_precision(X_test, y_test, n = 5):
    with tf.Graph().as_default() as g:

        image_shape = X_test[0].shape
        img_width = image_shape[0]
        img_height = image_shape[1]
        img_channels = image_shape[2]

        images_p = tf.placeholder(tf.float32, shape=[None, img_width, img_height, img_channels])
        labels_p = tf.placeholder(tf.int32, shape=[None])
        keep_prob_p = tf.placeholder(tf.float32)

        logits = classifier.inference(images_p, keep_prob_p)
        top_k_op = tf.nn.in_top_k(logits, labels_p, n)

        variable_averages = tf.train.ExponentialMovingAverage(
            classifier.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        precision = eval_once(saver, top_k_op, X_test, y_test,
                              images_p, labels_p, keep_prob_p)
        print('%s: test set precision %.3f' % (datetime.now(), precision))
        return precision

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

    idx = np.random.randint(len(X_test))
    img = X_test[idx]
    label = y_test[idx]

    evaluate_image(img, label)
    evaluate_precision(X_test, y_test)


if __name__ == '__main__':
    tf.app.run()
