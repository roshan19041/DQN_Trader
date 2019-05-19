#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:49:22 2019

@author: roshanprakash
"""
import tensorflow as tf

class Target_Network(tf.keras.models.Model):
    
    """ Deep Q-Network that uses 1-dimensional Convolution (temporal) and FC layers """

    def __init__(self, tau):
        """ 
        Initializes the target network.
        
        PARAMETERS
        ----------
        - tau : the learning rate for soft update of the network's weights
        
        RETURNS
        -------
        - None
        """
        super(Target_Network, self).__init__()
        self.tau = tau
        # placeholder(s) for input(s)
        self.x = tf.placeholder(shape=[None, 53], dtype=tf.float32)
        # model architecture : states --> fc(180) ---> fc(90) ---> fc(30) --> fc(3)/Q-values
        self.fc1 = tf.keras.layers.Dense(units=180, kernel_initializer=tf.keras.initializers.he_normal(seed=56789), activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=90, kernel_initializer=tf.keras.initializers.he_normal(seed=12345), activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(units=30, kernel_initializer=tf.keras.initializers.he_normal(seed=9999), activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=3, kernel_initializer=tf.initializers.truncated_normal(0, 0.01))
        # forward pass
        self.fc_1_out = self.fc1(self.x)
        self.fc_2_out = self.fc2(self.fc_1_out)
        self.fc_3_out = self.fc3(self.fc_2_out)
        self.model_out = self.out(self.fc_3_out)
           
    def soft_update(self, Q_Net, initialize=False):
        """ 
        Implements the weights update for this (target) network based on the prediction network's weights.
        
        PARAMETERS
        ----------
        - Q_Net : the prediction network used to adjust this network's weights(an instance of Q_Network)
        - initialize : if True, the network's weights is set to <Q_Net>'s weights
        
        RETURNS
        -------
        - None 
        """
        if initialize: # initialize the weights to be exactly the same as the other network
            tau = 1.0
        else:
            tau = self.tau
        prediction_network_weights = Q_Net.get_weights()
        target_network_weights = self.get_weights()
        for i in range(len(target_network_weights)):
            target_network_weights[i] = tau*prediction_network_weights[i]+\
                                (1-tau)*target_network_weights[i]
        # transfer weights
        self.set_weights(target_network_weights)