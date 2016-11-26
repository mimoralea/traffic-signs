from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_score, recall_score, f1_score

import pickle
import random

import tensorflow as tf
import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin

import time
import os

NUM_CLASSES = 43
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05       # Initial learning rate.

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', int(1e6),
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', 'train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', 'eval_dir',
                           """Direcotry where to write event logs.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def loss_func(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(
                               stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images, keep_prob):

    # conv1
    print("images", images.get_shape())
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 1, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    print("conv1", conv1.get_shape())

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    print("pool1", pool1.get_shape())

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    print("norm1", norm1.get_shape())

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    print("conv2", conv2.get_shape())

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    print("norm2", norm2.get_shape())

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print("pool2", pool2.get_shape())

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = np.prod(pool2.get_shape().as_list()[1:])
        reshape = tf.reshape(pool2, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)
    print("local3", local3.get_shape())

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
    print("local4", local4.get_shape())

    local4_drop = tf.nn.dropout(local4, keep_prob)
    print("local4_drop", local4_drop.get_shape())

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4_drop, weights), biases, name=scope.name)
        #softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    print("softmax_linear", softmax_linear.get_shape())

    softmax_normalized = tf.nn.softmax(softmax_linear)
    print("softmax_normalized", softmax_normalized.get_shape())
    return softmax_normalized


def train_func(total_loss, global_step):

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def augment_image(img):
    # img = tf.random_crop(img, [24, 24, 3])
    # img = tf.image.random_hue(img, 0.25)
    img = tf.image.random_saturation(img, lower=0, upper=10)
    img = tf.image.random_brightness(img, max_delta=0.8)
    img = tf.image.random_contrast(img, lower=0, upper=10)
    img = tf.image.per_image_whitening(img)
    return img

def augment_images(images, keep_prob_p):
    with tf.device('/cpu:0'):
        _, img_height, img_width, _ = images.get_shape().as_list()
        images = tf.cond(tf.equal(keep_prob_p, 1.0),
                         lambda: images,
                         lambda: tf.map_fn(
                             lambda img: augment_image(img), images))
        images = tf.image.rgb_to_grayscale(images)
        images = tf.image.resize_images(images, (img_height, img_width),
                                        method=0,
                                        align_corners=False)
        tf.image_summary(images.name, images, max_images=10)
    return images


def calculate_statistics(sess, images_p, labels_p, keep_prob_p,
                         tag_prefix, logits, data, y_true, sign_strings,
                         summary_writer, step):

    pred_prob = sess.run(logits,
                         feed_dict={images_p:data,
                                    labels_p:y_true,
                                    keep_prob_p:1.0})
    y_pred = np.argmax(pred_prob, axis=1)

    ck_score = cohen_kappa_score(y_true, y_pred, labels=range(NUM_CLASSES))
    c_report = classification_report(y_true, y_pred,
                                     labels=range(NUM_CLASSES),
                                     target_names=sign_strings['SignName'],
                                     sample_weight=None, digits=2)

    accuracy = jaccard_similarity_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=range(NUM_CLASSES), average='weighted')
    recall = recall_score(y_true, y_pred, labels=range(NUM_CLASSES), average='weighted')
    f1 = f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average='weighted')
    c_matrix = confusion_matrix(y_true, y_pred,
                                labels=range(NUM_CLASSES))

    accuracy_summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag_prefix + '_accuracy',
                                simple_value=accuracy)])
    precision_summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag_prefix + '_precision',
                                simple_value=precision)])
    recall_summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag_prefix + '_recall',
                                simple_value=recall)])
    f1_summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag_prefix + '_f1',
                                simple_value=f1)])
    ck_summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag_prefix + '_cohen_kappa',
                                simple_value=ck_score)])

    summary_writer.add_summary(accuracy_summary, step)
    summary_writer.add_summary(precision_summary, step)
    summary_writer.add_summary(recall_summary, step)
    summary_writer.add_summary(f1_summary, step)
    summary_writer.add_summary(ck_summary, step)

    return accuracy, precision, recall, f1, ck_score, c_report, c_matrix


def main(argv=None):  # pylint: disable=unused-argument

    training_file = 'data/train.p'
    testing_file = 'data/test.p'

    sign_strings = pd.read_csv('signnames.csv', index_col=0)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    idxs = np.array([], dtype=np.int32)
    for i in range(NUM_CLASSES):
        idxs = np.append(idxs, np.random.choice(
            np.where(y_train == i)[0], size=20, replace=False))
    X_validation, y_validation = X_train[idxs], y_train[idxs]
    X_train = np.delete(X_train, idxs, 0)
    y_train = np.delete(y_train, idxs, 0)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_p = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        labels_p = tf.placeholder(tf.int32, shape=[None])
        keep_prob_p = tf.placeholder(tf.float32)

        process_op = augment_images(images_p, keep_prob_p)

        logits = inference(process_op, keep_prob_p)
        loss = loss_func(logits, labels_p)

        train_op = train_func(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):

            start_time = time.time()
            batch_idxs = np.array([], dtype=np.int32)
            for i in range(NUM_CLASSES):
                batch_idxs = np.append(batch_idxs,
                                       np.random.choice(
                                           np.where(y_train == i)[0],
                                           size=100, replace=False))
            idx = np.random.choice(batch_idxs, size=FLAGS.batch_size, replace=False)
            sess.run(train_op,
                     feed_dict={images_p:X_train[idx],
                                labels_p:y_train[idx],
                                keep_prob_p:0.4})
            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                loss_value = sess.run(loss,
                                      feed_dict={images_p:X_train[idx],
                                                 labels_p:y_train[idx],
                                                 keep_prob_p:1.0})
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                summary_str = sess.run(summary_op,
                                       feed_dict={images_p:X_train[idx],
                                                  labels_p:y_train[idx],
                                                  keep_prob_p:1.0,
                                                  global_step:step})
                summary_writer.add_summary(summary_str, step)

                accuracy, precision, recall, f1, \
                    ck_score, c_report, c_matrix = calculate_statistics(
                        sess, images_p, labels_p, keep_prob_p, 'batch', logits,
                        X_train[idx], y_train[idx], sign_strings, summary_writer, step)
                format_str = ('batch train set | loss = %.2f, accuracy = %.2f, '
                              'precision = %.2f, recall = %.2f, f1 = %.2f, '
                              'cohen kappa = %.2f')
                print (format_str % (loss_value, accuracy, precision, recall, f1, ck_score))

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                train_idx = np.random.choice(len(X_train), 10000)
                accuracy, precision, recall, f1, \
                    ck_score, c_report, c_matrix = calculate_statistics(
                        sess, images_p, labels_p, keep_prob_p, 'train', logits,
                        X_train[train_idx], y_train[train_idx],
                        sign_strings, summary_writer, step)
                format_str = ('train data set | accuracy = %.2f, '
                              'precision = %.2f, recall = %.2f, f1 = %.2f, '
                              'cohen kappa = %.2f')
                print (format_str % (accuracy, precision, recall, f1, ck_score))

                accuracy, precision, recall, f1, \
                    ck_score, c_report, c_matrix = calculate_statistics(
                        sess, images_p, labels_p, keep_prob_p, 'validation', logits,
                        X_validation, y_validation, sign_strings, summary_writer, step)
                format_str = ('validation data set | accuracy = %.2f, '
                              'precision = %.2f, recall = %.2f, f1 = %.2f, '
                              'cohen kappa = %.2f')
                print (format_str % (accuracy, precision, recall, f1, ck_score))

            if step % 10000 == 0:
                with tf.variable_scope('conv1', reuse=True) as scope:
                    W_conv1 = tf.get_variable('weights', shape=[5, 5, 1, 64])
                    weights = W_conv1.eval(session=sess)
                    weights_path = os.path.join(FLAGS.checkpoint_dir,
                                                scope.name + '_weights_' + str(step) + '.npz')
                    with open(weights_path, "wb") as outfile:
                        np.save(outfile, weights)

                with tf.variable_scope('conv2', reuse=True) as scope:
                    W_conv2 = tf.get_variable('weights', shape=[5, 5, 64, 64])
                    weights = W_conv2.eval(session=sess)
                    weights_path = os.path.join(FLAGS.checkpoint_dir,
                                                scope.name + '_weights_' + str(step) + '.npz')
                    with open(weights_path, "wb") as outfile:
                        np.save(outfile, weights)

                accuracy, precision, recall, f1, \
                    ck_score, c_report, c_matrix = calculate_statistics(
                        sess, images_p, labels_p, keep_prob_p, 'test', logits,
                        X_test, y_test, sign_strings, summary_writer, step)
                format_str = ('test data set | accuracy = %.2f, '
                              'precision = %.2f, recall = %.2f, f1 = %.2f, '
                              'cohen kappa = %.2f')
                print (format_str % (accuracy, precision, recall, f1, ck_score))
                print(c_report)

                conf_matrix_path = os.path.join(
                    FLAGS.checkpoint_dir,
                    scope.name + '_cmatrix_' + str(step) + '.npz')
                with open(conf_matrix_path, "wb") as outfile:
                    np.save(outfile, c_matrix)


if __name__ == '__main__':
    tf.app.run()
