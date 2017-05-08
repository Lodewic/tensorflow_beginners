# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:13:47 2017

@author: Lodewic

This script is part of a series of Getting Started tutorials to learn Tensorflow.
Tutorial written along with https://www.tensorflow.org/get_started/input_fn

The goal is to learn to use custom input functions with tf.contrib.learn.
The data that is used is the Boston housing data from the UCI Housing Data Set.

More info on the dataset is found here: https://archive.ics.uci.edu/ml/datasets/Housing
"""

# Import __future__ functions 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Subfolder of the working directory where the dataset files are saved
# boston_train.csv : http://download.tensorflow.org/data/boston_train.csv
# boston_test.csv : http://download.tensorflow.org/data/boston_test.csv
# boston_predict.csv : http://download.tensorflow.org/data/boston_predict.csv
data_subfolder = "Boston_Housing/"


#%% Define column names and feature/label names
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
       "dis", "tax", "ptratio", "medv"]
FEATURES = COLUMNS[:-1]
LABEL = COLUMNS[-1]

#%% Load datasets from .csv files
training_set = pd.read_csv(data_subfolder + 'boston_train.csv',
                           skipinitialspace=True,
                           skiprows=1,
                           names=COLUMNS)
test_set = pd.read_csv(data_subfolder + 'boston_test.csv',
                       skipinitialspace=True,
                       skiprows=1,
                       names=COLUMNS)
prediction_set = pd.read_csv(data_subfolder+'boston_predict.csv',
                             skipinitialspace=True,
                             skiprows=1,
                             names=COLUMNS)

#%% Define the input function to be used
def input_fn(data_set):
    """
    Input function can be used to do preprocessing before feeding the data
    to our model. If our feature/label data is stored in a panda DF or numpy
    array, we need to convert it to Tensors before output
    
    The input function must return,
    feature_cols
        A dict that maps feature column names to (Sparse)Tensors
    labels
        A Tensor containing the label/target values
    """    
    # Preprocess your data and convert to Tensors
    # Specify the shape as well to avoid warnings in the original tutorial.
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1])
                        for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels

# Main function to train a simple neural network on the UCI Boston Housing Price
# dataset. Using tf.contrib.learn() and a custom input_fn()
def main():
    #%% Defining FeatureColumns
    feature_cols = [tf.contrib.layers.real_valued_column(k) 
                        for k in FEATURES]
    
    #%% Define regression model
    # A 2 layer neural network with 10,10 hidden nodes
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units = [10, 10],
                                                             model_dir='tmp/boston_model')
    
    #%% Train our model
    regressor.fit(input_fn=lambda: input_fn(training_set),
                   steps=5000)
    
    #%% Evaluate the model
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set),
                            steps=1)
    loss_score = ev['loss']
    print('Loss: {0:f}'.format(loss_score))
    
    #%% Make predictions with the model
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    predictions = list(itertools.islice(y, 6))
    print("\nPredictions = {}".format(str(predictions)))

if __name__=='__main__':
    main()