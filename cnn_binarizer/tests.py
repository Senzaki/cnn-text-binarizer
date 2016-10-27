#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from builtins import *

import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt

from cnn_binarizer import datasets

TRAINING_SAMPLES = 1
INPUT_SIZE = 100

def log_softmax(x):
    xdev = x - x.max(axis=1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=0)

class TestNetwork(object):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.input_var = T.tensor4('TestNetwork/input_var')
        self.labels_var = T.matrix('TestNetwork/labels_var')
        self.input_shape = (1, 1, INPUT_SIZE, INPUT_SIZE)
        self._create_net_structure()
        self._create_net_functions()

    def _create_net_structure(self):
        relu_initializer = lasagne.init.GlorotUniform(gain='relu')
        # Input layer
        in_lay = lasagne.layers.InputLayer(self.input_shape, input_var=self.input_var)
        # First set of encoding layers
        conv0_lay = lasagne.layers.Conv2DLayer(in_lay, 256, 5, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify)
        norm0_lay = lasagne.layers.BatchNormLayer(conv0_lay)
        relu0_lay = lasagne.layers.NonlinearityLayer(norm0_lay, nonlinearity=lasagne.nonlinearities.rectify)
        pool0_lay = lasagne.layers.MaxPool2DLayer(relu0_lay, 2, stride=2)
        # Second set of encoding layers
        conv1_lay = lasagne.layers.Conv2DLayer(pool0_lay, 128, 5, W=relu_initializer)
        norm1_lay = lasagne.layers.BatchNormLayer(conv1_lay)
        relu1_lay = lasagne.layers.NonlinearityLayer(norm1_lay, nonlinearity=lasagne.nonlinearities.rectify)
        pool1_lay = lasagne.layers.MaxPool2DLayer(relu1_lay, 2, stride=2) 
        # First set of decoding layers
        upsample1_lay = lasagne.layers.InverseLayer(pool1_lay, pool1_lay)
        deconv1_lay = lasagne.layers.InverseLayer(upsample1_lay, conv1_lay)
        # Second set of decoding layers
        upsample2_lay = lasagne.layers.InverseLayer(deconv1_lay, pool0_lay)
        deconv2_lay = lasagne.layers.InverseLayer(upsample2_lay, conv0_lay)
        # Classifying (softmax layer)
        classes_lay = lasagne.layers.Conv2DLayer(deconv2_lay, 2, 1)
        outshape_lay = lasagne.layers.ReshapeLayer(classes_lay, (2, INPUT_SIZE * INPUT_SIZE))
        shuffle_lay = lasagne.layers.DimshuffleLayer(outshape_lay, (1, 0))
        softmax_lay = lasagne.layers.NonlinearityLayer(shuffle_lay, nonlinearity=log_softmax)
        outshuffle_lay = lasagne.layers.DimshuffleLayer(softmax_lay, (1, 0))
        # softmax_lay = lasagne.layers.DenseLayer(in_lay, 2 * INPUT_SIZE * INPUT_SIZE, nonlinearity=lasagne.nonlinearities.sigmoid)
        # softmax_lay = lasagne.layers.ReshapeLayer(softmax_lay, (2, INPUT_SIZE * INPUT_SIZE))
        self.network = outshuffle_lay 

    def _create_net_functions(self):
        prediction = lasagne.layers.get_output(self.network)
#        loss = lasagne.objectives.categorical_crossentropy(prediction, self.labels_var).mean(dtype=theano.config.floatX)
#        loss = lasagne.objectives.squared_error(prediction, self.labels_var).mean(dtype=theano.config.floatX)
        loss = categorical_crossentropy_logdomain(prediction, self.labels_var).mean(dtype=theano.config.floatX)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.9)
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, self.labels_var).mean(dtype=theano.config.floatX)
        chosen_class = T.argmax(test_prediction, axis=1)
        well_predicted_class = T.eq(chosen_class, self.labels_var)
        test_accuracy = T.mean(well_predicted_class, dtype=theano.config.floatX)
        self.train_fn = theano.function([self.input_var, self.labels_var], loss, updates=updates)
        self.test_fn = theano.function([self.input_var, self.labels_var], [test_loss, test_accuracy])
        # output_image = T.reshape(T.argmax(test_prediction, axis=0), (INPUT_SIZE, INPUT_SIZE))
        output_image = T.exp(T.reshape(test_prediction, (2, INPUT_SIZE, INPUT_SIZE)))
        self.forward_fn = theano.function([self.input_var], output_image)

    def train(self, input, labels, num_epochs=10):
        print('Starting training...')
        for epoch in range(num_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1, num_epochs))
            for i in range(TRAINING_SAMPLES):
                loss = self.train_fn(input[[i], :, :, :], labels[i, :, :])
                print('Loss:', loss)
        for num_test in range(1):
            i = np.random.randint(TRAINING_SAMPLES)
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(labels[i, :, :].argmax(axis=0).reshape((INPUT_SIZE, INPUT_SIZE)))
            plt.colorbar()
            prediction = self.forward_fn(input[[i], :, :, :])[1, :, :]
            plt.subplot(2, 1, 2)
            plt.imshow(prediction)
            plt.colorbar()
            plt.show()
            print(prediction)
            print(labels[i, :, :].argmax(axis=0).reshape((INPUT_SIZE, INPUT_SIZE)))
        return loss

    def test_stats(self, input, labels):
        return self.test_fn(input, labels)

    def forward(self, input):
        return self.forward_fn(input)

def run_testnetwork_test(dataset):
    network = TestNetwork()
#    train_input = np.random.rand(1, 1, INPUT_SIZE, INPUT_SIZE).astype(theano.config.floatX)
#    train_labels = np.random.rand(2, INPUT_SIZE * INPUT_SIZE).astype(dtype=theano.config.floatX)
    train_input, train_labels_indices = dataset.create_training_arrays(TRAINING_SAMPLES, (INPUT_SIZE, INPUT_SIZE))
    train_labels_indices = np.reshape(train_labels_indices, (TRAINING_SAMPLES, INPUT_SIZE * INPUT_SIZE))
    train_labels = np.zeros((TRAINING_SAMPLES, 2, INPUT_SIZE * INPUT_SIZE), dtype=theano.config.floatX)
    for sample_idx in range(TRAINING_SAMPLES):
        for pixel_idx in range(INPUT_SIZE * INPUT_SIZE):
            train_labels[sample_idx, int(train_labels_indices[sample_idx, pixel_idx]), pixel_idx] = 1
    print(train_labels)
    final_loss = network.train(train_input, train_labels)
    print('Final loss:', final_loss)
    return network

def load_datasets():
    return datasets.TrainingDataSet('data/train/input', 'data/train/labels')

def run_all_tests():
    dataset = load_datasets()
    run_testnetwork_test(dataset)

if __name__ == '__main__':
    run_all_tests()

