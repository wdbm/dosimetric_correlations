#!/usr/bin/env python

"""
################################################################################
#                                                                              #
# cures_cancer                                                                 #
#                                                                              #
################################################################################
#                                                                              #
# LICENCE INFORMATION                                                          #
#                                                                              #
# This program is a neural network.                                            #
#                                                                              #
# copyright (C) 2017 William Breaden Madden, Gavin Kirby                       #
#                                                                              #
# This software is released under the terms of the GNU General Public License  #
# version 3 (GPLv3).                                                           #
#                                                                              #
# This program is free software: you can redistribute it and/or modify it      #
# under the terms of the GNU General Public License as published by the Free   #
# Software Foundation, either version 3 of the License, or (at your option)    #
# any later version.                                                           #
#                                                                              #
# This program is distributed in the hope that it will be useful, but WITHOUT  #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     #
# more details.                                                                #
#                                                                              #
# For a copy of the GNU General Public License, see                            #
# <http://www.gnu.org/licenses/>.                                              #
#                                                                              #
################################################################################

usage:
    program [options]

options:
    -h, --help                        display help message
    --version                         display version and exit

    --epochs=INT                      number of training epochs     [default: 700001]
    --learning_rate=FLOAT             learning rate                 [default: 0.09]
    --number_nodes_layer=INT          number of nodes per layer     [default: 50]
    --dropout_keep_probability=FLOAT  dropout keep probability      [default: 0.7]

    --number_targets=INT              number of target variables for model (number of
                                      rightmost columns in CSV that are output variables)
                                                                    [default: 3]

    --test_set_fraction=FLOAT         fraction of data for testing  [default: 0.33]

    --infile=FILENAME                 CSV input file                [default: data_preprocessed.csv]

    --save_results_to_file            save results to file
    --results_file=FILENAME           CSV results file              [default: results.csv]

    --TensorBoard                     run with TensorBoard
    --path_logs=PATH                  TensorBoard logs path         [default: /tmp/run]
"""

from __future__ import division
import datetime
import docopt
import os
import os.path
import subprocess

import numpy as np
import sklearn.model_selection
import tensorflow as tf

name    = "cures_cancer"
version = "2017-07-30T2238Z"
logo    = None

def main(options):

    # configuration
    number_targets           = int(options["--number_targets"])
    epochs                   = int(options["--epochs"])
    learning_rate            = float(options["--learning_rate"])
    number_nodes_layer       = int(options["--number_nodes_layer"])
    dropout_keep_probability = float(options["--dropout_keep_probability"])
    fraction_test_set        = float(options["--test_set_fraction"])
    path_logs                = options["--path_logs"]
    use_TensorBoard          = options["--TensorBoard"]
    save_results_to_file     = options["--save_results_to_file"]
    filename_CSV_input       = options["--infile"]
    filename_results         = options["--results_file"]

    if not os.path.isfile(os.path.expandvars(filename_CSV_input)):
        print("file {filename} not found".format(
            filename = filename_CSV_input
        ))
        exit()

    tf.reset_default_graph()

    # TensorBoard
    if use_TensorBoard:
        subprocess.Popen(["killall tensorboard"],                                            shell = True)
        subprocess.Popen(["rm -rf {path_logs}".format(path_logs = path_logs)],               shell = True)
        subprocess.Popen(["tensorboard --logdir={path_logs}".format(path_logs = path_logs)], shell = True)
        subprocess.Popen(["xdg-open http://127.0.1.1:6006"],                                 shell = True)

    data = np.loadtxt(
        filename_CSV_input,
        skiprows  = 1,
        delimiter = ",",
        dtype     = np.float32
    )

    x_data = data[:, 0:- number_targets]
    y_data = data[:, - number_targets:]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_data, y_data, test_size = fraction_test_set, random_state = 42
    )

    with tf.name_scope("input"):
        X          = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        Y          = tf.placeholder(tf.float32, [None, y_train.shape[1]])
    tf.summary.histogram("input", X)

    with tf.name_scope("architecture"):
        W1         = tf.Variable(tf.random_normal([x_train.shape[1], number_nodes_layer]),   name = "weight1")
        b1         = tf.Variable(tf.random_normal([number_nodes_layer]),                     name = "bias1"  )
        layer1     = tf.sigmoid(tf.matmul(X, W1) + b1)
        layer1     = tf.nn.dropout(layer1, keep_prob = dropout_keep_probability)

        W2         = tf.Variable(tf.random_normal([number_nodes_layer, number_nodes_layer]), name = "weight2")
        b2         = tf.Variable(tf.random_normal([number_nodes_layer]),                     name = "bias2"  )
        layer2     = tf.sigmoid(tf.matmul(layer1, W2) + b2)
        layer2     = tf.nn.dropout(layer2, keep_prob = dropout_keep_probability)

        W3         = tf.Variable(tf.random_normal([number_nodes_layer, number_nodes_layer]), name = "weight3")
        b3         = tf.Variable(tf.random_normal([number_nodes_layer]),                     name = "bias3"  )
        layer3     = tf.sigmoid(tf.matmul(layer2, W3) + b3)
        layer3     = tf.nn.dropout(layer3, keep_prob = dropout_keep_probability)

        W4         = tf.Variable(tf.random_normal([number_nodes_layer, y_train.shape[1]]),   name = "weight4")
        b4         = tf.Variable(tf.random_normal([y_train.shape[1]]),                       name = "bias4"  )
        #hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
        hypothesis = tf.matmul(layer3, W4) + b4
    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("layer1", layer1)
    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("layer2", layer2)
    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("layer3", layer3)
    tf.summary.histogram("W4", W4)
    tf.summary.histogram("b4", b4)
    tf.summary.histogram("hypothesis", hypothesis)

    with tf.name_scope("cost"):
        #cost       = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
        #cost       = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = hypothesis))
        cost       = tf.reduce_mean(tf.square(hypothesis - Y))
        #train      = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
        train      = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    tf.summary.scalar("cost", cost)

    with tf.name_scope("accuracy"):
        #accuracy    = tf.subtract(tf.constant(100, dtype = tf.float32), tf.multiply(tf.divide(tf.sqrt(tf.square(hypothesis - Y)), Y), tf.constant(100, dtype = tf.float32)))
        accuracy    = tf.divide(hypothesis, Y)
    tf.summary.histogram("accuracy", accuracy)

    summary_operation = tf.summary.merge_all()

    with tf.Session() as sess:

        time_start = datetime.datetime.utcnow()

        sess.run(tf.global_variables_initializer())

        if use_TensorBoard:
            writer = tf.summary.FileWriter(path_logs)

        for step in range(epochs):

            _, summary = sess.run([train, summary_operation], feed_dict = {X: x_train, Y: y_train})

            if use_TensorBoard:
                writer.add_summary(summary, step)

            if step % 100 == 0:
                print("\nstep: {step}\ncost: {cost}".format(
                    step = step,
                    cost = sess.run(cost, feed_dict = {X: x_train, Y: y_train})
                ))

        print("\naccuracy report:")
        h, a = sess.run([hypothesis, accuracy], feed_dict = {X: x_test, Y: y_test})
        print("\ntest features data:\n\n{x_test}".format(
            x_test = x_test
        ))
        print("\ntest targets data:\n\n{y_test}".format(
            y_test = y_test
        ))
        print("\nhypothesis (predicted values):\n\n{hypothesis}\n\naccuracy "
              "(hypothesis / data -- i.e. closer to unity is more accurate):\n\n{accuracy}".format(
            hypothesis = h,
            accuracy   = a
        ))

        time_stop = datetime.datetime.utcnow()

        print("\naccuracy mean:               {accuracy_mean}\naccuracy standard deviation: {accuracy_standard_deviation}".format(
            accuracy_mean               = a.mean(),
            accuracy_standard_deviation = a.std()
        ))

    if save_results_to_file:
        if not os.path.isfile(filename_results):
            with open(filename_results, "a") as file_results:
                line = [
                    "duration_hours",
                    "epochs",
                    "learning_rate",
                    "number_nodes_layer",
                    "dropout_keep_probability",
                    "accuracy_mean",
                    "accuracy_standard_deviation"
                ]
                file_results.write(",".join(line) + "\n")

        with open(filename_results, "a") as file_results:
            line = [
                str((time_stop - time_start).seconds / 3600),
                str(epochs),
                str(learning_rate),
                str(number_nodes_layer),
                str(dropout_keep_probability),
                str(a.mean()),
                str(a.std())
            ]
            file_results.write(",".join(line) + "\n")

    if use_TensorBoard:
        subprocess.Popen(["killall tensorboard"],                              shell = True)
        subprocess.Popen(["rm -rf {path_logs}".format(path_logs = path_logs)], shell = True)

if __name__ == "__main__":
    options = docopt.docopt(__doc__)
    if options["--version"]:
        print(version)
        exit()
    main(options)
