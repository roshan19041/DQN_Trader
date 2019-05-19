#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:50:32 2019

@author: roshanprakash
"""

import tensorflow as tf

class Prediction_Network(tf.keras.models.Model):
    
    """ Deep Q-Network. """
    
    def __init__(self, learning_rate, discount_rate):
        """
        Initializes the network.
        
        PARAMETERS
        ----------
        - learning_rate(float) : the learning rate used while updating the network's weights
        - discount_rate(float) : the agent's discount rate (used while computing loss)
       
        RETURNS
        -------
        - None 
        """
        super(Prediction_Network, self).__init__()
        self.learning_rate = learning_rate
        self.discount = discount_rate
        # placeholders for inputs
        self.x = tf.placeholder(shape=[None, 53], dtype=tf.float32)
        self.a = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.r = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.Q_t = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # model architecture : states --> fc(180) ---> fc(90) ---> fc(30) --> fc(3)/Q-values
        self.fc1 = tf.keras.layers.Dense(units=180, kernel_initializer=tf.initializers.identity(), activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=90, kernel_initializer=tf.initializers.identity(), activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(units=30, kernel_initializer=tf.initializers.identity(), activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=3, kernel_initializer=tf.initializers.truncated_normal(0, 0.01))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # forward pass
        self.fc_1_out = self.fc1(self.x)
        self.fc_2_out = self.fc2(self.fc_1_out)
        self.fc_3_out = self.fc3(self.fc_2_out)
        self.model_out = self.out(self.fc_3_out)
        self.Q_p = tf.batch_gather(self.model_out, self.a)
        # loss computation and weight updates
        self.loss = tf.reduce_mean(tf.square(tf.subtract(tf.add(self.r, tf.scalar_mul(self.discount, self.Q_t)), self.Q_p)))
        self.train_op = self.optimizer.minimize(self.loss)