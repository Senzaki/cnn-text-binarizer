#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from builtins import *

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, InverseLayer, ReshapeLayer, NonlinearityLayer, batch_norm
import matplotlib.pyplot as plt

from cnn_binarizer import datasets

TRAINING_BATCHES = 300
BATCH_SIZE = 4
TOTAL_TRAINING_SAMPLES = TRAINING_BATCHES * BATCH_SIZE
INPUT_SIZE = 200

def log_softmax(x):
    xdev = x - x.max(axis=1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=0)

class TestNetwork(object):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.input_var = T.tensor4('TestNetwork/input_var')
        self.labels_var = T.tensor3('TestNetwork/labels_var')
        self.input_shape = (BATCH_SIZE, 1, INPUT_SIZE, INPUT_SIZE)
        self._create_net_structure()
        self._create_net_functions()

    def _create_net_structure(self):
        linear_initializer = lasagne.init.GlorotUniform()
        relu_initializer = lasagne.init.GlorotUniform(gain='relu')
        # Input layer
        in_lay = InputLayer(self.input_shape, input_var=self.input_var)
        # First set of encoding layers
        conv0_lay = batch_norm(Conv2DLayer(in_lay, 64, 7, pad=3, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify))
        pool0_lay = MaxPool2DLayer(conv0_lay, 2, stride=2)
        # Second set of encoding layers
        conv1_lay = batch_norm(Conv2DLayer(pool0_lay, 64, 7, pad=3, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify))
        pool1_lay = MaxPool2DLayer(conv1_lay, 2, stride=2) 
        # Third set of encoding layers
        conv2_lay = batch_norm(Conv2DLayer(pool1_lay, 64, 7, pad=3, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify))
        pool2_lay = MaxPool2DLayer(conv2_lay, 2, stride=2) 
        dropenc2_lay = DropoutLayer(pool2_lay, p=0.5)
        # Fourth set of encoding layers
        conv3_lay = batch_norm(Conv2DLayer(dropenc2_lay, 64, 7, pad=3, W=relu_initializer, nonlinearity=lasagne.nonlinearities.rectify))
        pool3_lay = MaxPool2DLayer(conv3_lay, 2, stride=2) 
        dropenc3_lay = DropoutLayer(pool3_lay, p=0.5)
        # First set of decoding layers
        upsample3_lay = InverseLayer(dropenc3_lay, pool3_lay)
        deconv3_lay = batch_norm(Conv2DLayer(upsample3_lay, 64, 7, pad=3, W=linear_initializer, nonlinearity=None))
        dropdec3_lay = DropoutLayer(deconv3_lay, p=0.5)
        # Second set of decoding layers
        upsample2_lay = InverseLayer(dropdec3_lay, pool2_lay)
        deconv2_lay = batch_norm(Conv2DLayer(upsample2_lay, 64, 7, pad=3, W=linear_initializer, nonlinearity=None))
        dropdec2_lay = DropoutLayer(deconv2_lay, p=0.5)
        # Third set of decoding layers
        upsample1_lay = InverseLayer(dropdec2_lay, pool1_lay)
        deconv1_lay = batch_norm(Conv2DLayer(upsample1_lay, 64, 7, pad=3, W=linear_initializer, nonlinearity=None))
        # Fourth set of decoding layers
        upsample0_lay = InverseLayer(deconv1_lay, pool0_lay)
        deconv0_lay = batch_norm(Conv2DLayer(upsample0_lay, 64, 7, pad=3, W=linear_initializer, nonlinearity=None))
        # Classifying (softmax layer)
        classes_lay = Conv2DLayer(deconv0_lay, 2, 1)
        outshape_lay = ReshapeLayer(classes_lay, (BATCH_SIZE, 2, INPUT_SIZE * INPUT_SIZE))
        softmax_lay = NonlinearityLayer(outshape_lay, nonlinearity=log_softmax)
        self.network = softmax_lay 

    def _create_net_functions(self):
        prediction = lasagne.layers.get_output(self.network)
#        loss = lasagne.objectives.categorical_crossentropy(prediction, self.labels_var).mean(dtype=theano.config.floatX)
#        loss = lasagne.objectives.squared_error(prediction, self.labels_var).mean(dtype=theano.config.floatX)
        loss = categorical_crossentropy_logdomain(prediction, self.labels_var).mean(dtype=theano.config.floatX)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.9)
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = categorical_crossentropy_logdomain(test_prediction, self.labels_var).mean(dtype=theano.config.floatX)
        chosen_class = T.argmax(test_prediction, axis=1)
        well_predicted_class = T.eq(chosen_class, self.labels_var)
        test_accuracy = T.mean(well_predicted_class, dtype=theano.config.floatX)
        self.train_fn = theano.function([self.input_var, self.labels_var], loss, updates=updates)
        self.test_fn = theano.function([self.input_var, self.labels_var], [test_loss, test_accuracy])
        # output_image = T.reshape(T.argmax(test_prediction, axis=0), (INPUT_SIZE, INPUT_SIZE))
        output_image = T.exp(test_prediction)
        self.forward_fn = theano.function([self.input_var], output_image)

    def train(self, input, labels, num_epochs=1):
        print('Starting training...')
        for epoch in range(num_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1, num_epochs))
            for batch_idx in range(TRAINING_BATCHES):
                batch_input = input[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE, :, :, :]
                batch_labels = labels[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE, :, :]
                loss = self.train_fn(batch_input, batch_labels)
                print('Loss:', loss)
        for num_test in range(50):
            i = np.random.randint(TOTAL_TRAINING_SAMPLES - BATCH_SIZE + 1)
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(input[i, :, :, :].reshape((INPUT_SIZE, INPUT_SIZE)))
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(labels[i, :, :].argmax(axis=0).reshape((INPUT_SIZE, INPUT_SIZE)))
            plt.colorbar()
            prediction = self.forward_fn(input[i:i + BATCH_SIZE, :, :, :])[0, :, :].reshape((2, INPUT_SIZE, INPUT_SIZE))
            plt.subplot(2, 2, 3)
            plt.imshow(prediction[1, :])
            plt.colorbar()
            plt.subplot(2, 2, 4)
            plt.imshow(prediction.argmax(axis=0))
            plt.colorbar()
            plt.show()
        return loss

    def test_stats(self, input, labels):
        return self.test_fn(input, labels)

    def forward(self, input):
        return self.forward_fn(input)

def create_labels_from_images(images):
    choices = np.eye(2, dtype=theano.config.floatX).reshape((2, 2, 1))
    num_images = images.shape[0]
    num_pixels = images.shape[2] * images.shape[3]
    indices = images.reshape((num_images, 1, num_pixels)).astype(int)
    return np.choose(indices, choices)

def run_testnetwork_test(dataset):
    print('Preparing network...')
    network = TestNetwork()
#    train_input = np.random.rand(1, 1, INPUT_SIZE, INPUT_SIZE).astype(theano.config.floatX)
#    train_labels = np.random.rand(2, INPUT_SIZE * INPUT_SIZE).astype(dtype=theano.config.floatX)
    print('Preparing input...')
    train_input, train_labels_images = dataset.create_training_arrays(TOTAL_TRAINING_SAMPLES, (INPUT_SIZE, INPUT_SIZE))
    train_labels = create_labels_from_images(train_labels_images)
    final_loss = network.train(train_input, train_labels, 5)
    print('Final loss:', final_loss)
    return network

def load_datasets():
    return datasets.TrainingDataSet('data/train/input', 'data/train/labels')

def run_all_tests():
    dataset = load_datasets()
    run_testnetwork_test(dataset)

if __name__ == '__main__':
    run_all_tests()

