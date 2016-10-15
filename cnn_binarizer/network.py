#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import lasagne

class Network(object):
    def __init__(self, input_layer_shape, kernel_size, num_conv_layers, pooling_size, name='Network', learning_rate=0.1):
        super(Network, self).__init__()
        self.name = name
        self.input_var = T.tensor4(self.name + '.input')
        self.label_var = T.matrix(self.name + '.labels')
        self.input_layer = lasagne.layers.InputLayer(input_layer_shape, input_var=self.input_var)
        prev_layer = self.input_layer
        relu_initializer = lasagne.init.GlorotUniform(gain='relu')
        pooling_layers = list()
        for layer_id in range(num_conv_layers):
            #TODO: Change amount of conv filter per layer?
            prev_layer = Network._create_reducing_layer(prev_layer, kernel_size, pooling_size, relu_initializer)
            pooling_layers.append(prev_layer)
        for pooling_layer in reversed(pooling_layers):
            prev_layer = Network._create_increasing_layer(prev_layer, pooling_layer, kernel_size, relu_initializer)
        self.output_layer = lasagne.layers.DenseLayer(prev_layer, 2, nonlinearity=lasagne.nonlinearities.softmax)
        output = lasagne.layers.get_output(self.output_layer)
        loss = lasagne.objectives.categorical_crossentropy(output, self.label_var).mean()
        self.params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params=self.params, learning_rate=learning_rate, momentum=0.9)
        self.train = theano.function([self.input_var, self.label_var], loss, updates=updates)

    @staticmethod
    def _create_reducing_layer(prev_layer, kernel_size, pooling_size, relu_initializer):
        conv_layer = lasagne.layers.Conv2DLayer(prev_layer, 1, kernel_size, nonlinearity=lasagne.nonlinearities.rectify, W=relu_initializer)
        batch_normalization_layer = lasagne.layers.BatchNormLayer(conv_layer)
        pooling_layer = lasagne.layers.MaxPool2DLayer(batch_normalization_layer, pooling_size)
        return pooling_layer

    @staticmethod
    def _create_increasing_layer(prev_layer, pooling_layer, kernel_size, relu_initializer):
        unpooling_layer = lasagne.layers.InverseLayer(prev_layer, pooling_layer)
        unconv_layer = lasagne.layers.Conv2DLayer(unpooling_layer, 1, kernel_size, nonlinearity=lasagne.nonlinearities.rectify, W=relu_initializer)
        batch_normalization_layer = lasagne.layers.BatchNormLayer(unconv_layer)
        return batch_normalization_layer

