# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:27:13 2017

@author: Lodewic
Copying the lines from https://www.tensorflow.org/get_started/mnist/pros
Expect to see around a 92% accuracy using a softmax regression model
"""
# Use the MNIST data set provided by the tensorflow package
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import tensorflow
# InteractiveSession() for tutorial purpose
import tensorflow as tf
sess = tf.InteractiveSession()

#%% Model initialization
# Define hyperparameters
learning_rate = 0.5
num_iterations = 1000
batch_size = 100

# Placeholders for image and label data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define variables such as weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize all variables at once
sess.run(tf.global_variables_initializer())

# Regression model
y = tf.matmul(x, W) + b

# Define loss function
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#%% Training
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

for _ in range(num_iterations):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_ : batch[1]})
   
#%% Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#if 'session' in locals() and session is not None:
#    print('Close interactive session')
#    session.close()