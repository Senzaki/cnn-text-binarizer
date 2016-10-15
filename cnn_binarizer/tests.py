#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne

class TestNetwork(object):
    def __init__(self, input_var, labels_var):
        self.input_shape = (1, 1, 110, 110)
        relu_initializer = lasagne.init.GlorotUniform(gain='relu')
        # Input layer
        in_lay = lasagne.layers.InputLayer(self.input_shape, input_var=input_var)
        # First set of conv layers
        conv0_lay = lasagne.layers.Conv2DLayer(in_lay, 32, 7, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify)
        norm0_lay = lasagne.layers.BatchNormLayer(conv0_lay)
        relu0_lay = lasagne.layers.NonlinearityLayer(norm0_lay, nonlinearity=lasagne.nonlinearities.rectify)
        pool0_lay = lasagne.layers.MaxPool2DLayer(relu0_lay, 2)
        # Second set of conv layers
        conv1_lay = lasagne.layers.Conv2DLayer(pool0_lay, 32, 7, W=relu_initializer)
        norm1_lay = lasagne.layers.BatchNormLayer(conv1_lay)
        relu1_lay = lasagne.layers.NonlinearityLayer(norm1_lay, nonlinearity=lasagne.nonlinearities.rectify)
        pool1_lay = lasagne.layers.MaxPool2DLayer(relu1_lay, 2, stride=2) 
        # First set of unconv layers
        unpool1_lay = lasagne.layers.InverseLayer(pool1_lay, pool1_lay)
        # Classifying (softmax layer)
        self.network = softmax_lay

def run_test_network_test():
    network = create_test_network()

def run_all_tests():
    test_network_test()

if __name__ == '__main__':
    run_all_tests()

