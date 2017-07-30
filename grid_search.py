#!/usr/bin/env python

"""
################################################################################
#                                                                              #
# grid_search                                                                  #
#                                                                              #
################################################################################
#                                                                              #
# LICENCE INFORMATION                                                          #
#                                                                              #
# This program prepares a hyperparameter grid search.                          #
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
"""

import docopt
import itertools
import os
import os.path
import subprocess

import numpy as np
import sklearn.model_selection
import tensorflow as tf

name    = "grid_search"
version = "2017-07-30T2238Z"
logo    = None

def main():

    epochs                   = [500, 1000, 5000, 10000, 100000]
    learning_rate            = [0.07, 0.08, 0.09]
    number_nodes_layer       = [50, 100, 150, 200]
    dropout_keep_probability = [1.0]
    element_specifications   = [epochs, learning_rate, number_nodes_layer, dropout_keep_probability]

    combinations = [list(list_configuration) for list_configuration in list(itertools.product(*element_specifications))]

    commands = ["#!/bin/bash"]
    for combination in combinations:
        command = "./cures_cancer.py --epochs={epochs} --learning_rate={learning_rate} --number_nodes_layer={number_nodes_layer} --dropout_keep_probability={dropout_keep_probability} --save_results_to_file".format(
            epochs                   = combination[0],
            learning_rate            = combination[1],
            number_nodes_layer       = combination[2],
            dropout_keep_probability = combination[3]
        )
        commands.append(command)

    for command in commands:
        print(command)

if __name__ == "__main__":
    main()
