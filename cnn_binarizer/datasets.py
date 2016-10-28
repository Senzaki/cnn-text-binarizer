#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from builtins import *

from PIL import Image
from os import walk
from os.path import relpath
from os.path import splitext
import numpy as np
import theano
from math import floor
import sys

DEFAULT_SPLIT_RATIO = 0.8

class DataSet(object):
    def __init__(self, folder):
        super(DataSet, self).__init__()
        self.folder = folder
        self.imagespaths = {}

    def preload(self):
        """Finds all usable images in the path."""
        for (path, _, files) in walk(self.folder):
            path = relpath(path, self.folder)
            for filename in files:
                if path != '.':
                    filename = path + '/' + filename
                (name, _) = splitext(filename)
                if name in self.imagespaths:
                    print('Warning: ' + name + ' corresponds to several images in the dataset.')
                self.imagespaths[name] = self.folder + '/' + filename

    def images_names(self):
        """
        Returns:
            List[str]: Preloaded images names.
        """
        return self.imagespaths.keys()

    def open_image(self, imagename):
        """
        Opens an image.

        Args:
            imagename (str): Name of the image, as returned by images_names (relative path without the extension).

        Returns:
            PIL.Image: Loaded image, or None if the image could not be loaded.
        """
        try:
            return Image.open(self.imagespaths[imagename])
        except KeyError:
            print('Unknown image name: ' + imagename)
        except IOError:
            print('I/O error while opening image: ' + imagespaths[imagename])
        return None

    def unload(self):
        """Unloads found images names."""
        self.imagespaths = {}

    def common_images(self, other):
        """
        Find common images between two data sets (=same name).

        Args:
            other (DataSet): Other data set.
        
        Returns
            Set[str]: Names of all the common images.
        """
        return self.imagespaths.keys() & other.imagespaths.keys()

class DataSetSplitter(object):
    def __init__(self, input_dataset, labels_dataset, training_ratio=DEFAULT_SPLIT_RATIO):
        super(DataSetSplitter, self).__init__()
        if not input_dataset.imagespaths:
            raise ValueError('Input dataset is empty or not preloaded.')
        if not labels_dataset.imagespaths:
            raise ValueError('Labels dataset is empty or not preloaded.')
        self.valid_names = list(input_dataset.common_images(labels_dataset))
        self.new_split(training_ratio)

    def new_split(self, training_ratio=DEFAULT_SPLIT_RATIO):
        shuffled_names = np.array(self.valid_names, dtype=str)
        np.random.shuffle(shuffled_names)
        split_index = floor(len(self.valid_names) * training_ratio)
        self.training_names = shuffled_names[:split_index]
        self.validation_names = shuffled_names[split_index + 1:]

class TrainingDataSet(object):
    def __init__(self, input_folder, labels_folder, training_ratio=DEFAULT_SPLIT_RATIO):
        super(TrainingDataSet, self).__init__()
        self.input_dataset = DataSet(input_folder)
        self.input_dataset.preload()
        self.labels_dataset = DataSet(labels_folder)
        self.labels_dataset.preload()
        self.splitter = DataSetSplitter(self.input_dataset, self.labels_dataset, training_ratio)

    def create_training_arrays(self, num_samples, samples_size, dtype=theano.config.floatX):
        remaining_images = self.splitter.training_names.size
        current_array_index = 0
        if len(samples_size) != 2:
            raise ValueError('samples_size must contain 2 values.')
        input_array = np.empty((num_samples, 1, samples_size[0], samples_size[1]), dtype=dtype)
        labels_array = np.empty(input_array.shape, dtype=dtype)
        shuffled_indices = np.random.permutation(num_samples)
        for image_name in self.splitter.training_names:
            if current_array_index != num_samples:
                with self.input_dataset.open_image(image_name).convert('F') as input_image, self.labels_dataset.open_image(image_name).convert('F') as label_image:
                    num_samples_for_image = (num_samples - current_array_index) // remaining_images
                    (w, h) = input_image.size
                    if (w, h) != label_image.size:
                        raise ValueError('Input image and label image for ' + image_name + ' do not have the same size.')
                    if w < samples_size[0] or h < samples_size[1]:
                        print('Warning: Discarding image ' + image_name + ' because it is smaller than the sample size (this might create errors if it is the last image).', file=sys.stderr)
                    else:
                        for patch_index in range(num_samples_for_image):
                            x = np.random.randint(0, w - samples_size[0])
                            y = np.random.randint(0, h - samples_size[1])
                            shuffled_index = shuffled_indices[current_array_index]
                            rect = (x, y, x + samples_size[0], y + samples_size[1])
                            input_array[shuffled_index, 0, :, :] = np.asarray(input_image.crop(rect), dtype=dtype)
                            labels_array[shuffled_index, 0, :, :] = np.asarray(label_image.crop(rect), dtype=dtype)
                            current_array_index += 1
                remaining_images -= 1
        if current_array_index != num_samples:
            raise RuntimeError('Not enough samples were generated (not enough valid images?)')
        return input_array / 255, labels_array / 255



