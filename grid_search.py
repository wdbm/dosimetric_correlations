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
# This program is prepares hyperparameter grid search.                         #
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
version = "2017-07-27T1717Z"
logo    = None

def main():

    epochs                 = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 20000]
    learning_rate          = [0.005, 0.01, 0.015, 0.02, 0.07, 0.08]
    number_nodes_layer     = [180, 200, 220, 250, 300]
    element_specifications = [epochs, learning_rate, number_nodes_layer]

    combinations = [list(list_configuration) for list_configuration in list(itertools.product(*element_specifications))]

    commands = []
    for combination in combinations:
        command = "./cures_cancer.py --epochs={epochs} --learning_rate={learning_rate} --number_nodes_layer={number_nodes_layer} --save_results_to_file".format(
            epochs             = combination[0],
            learning_rate      = combination[1],
            number_nodes_layer = combination[2]
        )
        commands.append(command)

    for command in commands:
        print(command)

if __name__ == "__main__":
    main()
