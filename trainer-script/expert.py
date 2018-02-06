from __future__ import absolute_import, unicode_literals
from tensorflow.python.tools import freeze_graph
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import os

mnist = input_data.read_data_sets("./data/", one_hot=True)

g = tf.Graph()
with g.as_default():
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(x_image, kernel_size=[5, 5], filters=32, activation=tf.nn.relu)
    max_pooling1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

    conv2 = tf.layers.conv2d(max_pooling1, kernel_size=[5, 5], filters=64, activation=tf.nn.relu)
    max_pooling2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

    flatten = tf.layers.flatten(max_pooling1)
    fc1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)

    keep_prob = tf.placeholder("float")
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    logits = tf.layers.dense(fc1_drop, 10)

    prob = tf.nn.softmax(logits, name='y_')

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(500):
            batch = mnist.train.next_batch(64)
            if i % 100 == 0:
                train_accuracy = accuracy.eval({x: batch[0], y_: batch[1], keep_prob: 1.0}, sess)
                print("step {}, training accuracy {}".format(i, train_accuracy))
            train_step.run({x: batch[0], y_: batch[1], keep_prob: 0.5}, sess)

        print("test accuracy {}".format(
            accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, sess)))

        dir_name = './tmp'
        tf.train.write_graph(sess.graph_def, '.', os.path.join(dir_name, 'mnist.pbtxt'))

        saver = tf.train.Saver()
        saver.save(sess, save_path=os.path.join(dir_name, 'mnist.ckpt'))

        input_graph_path = os.path.join(dir_name, 'mnist.pbtxt')
        ckpt_path = os.path.join(dir_name, 'mnist.ckpt')
        output_frozen_graph_name = 'frozen_mnist.pb'

        freeze_graph.freeze_graph(input_graph_path, input_saver='', input_binary=False,
                                  input_checkpoint=ckpt_path, output_node_names='y_',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0', output_graph=output_frozen_graph_name,
                                  clear_devices=True, initializer_nodes='')
