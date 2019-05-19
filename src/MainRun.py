#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:46:37 2019

@author: roshanprakash
"""
from PredictionNetwork import *
from TargetNetwork import *
from Trader import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main_run(data_path, epsilon=0.5, decay=1e-7, alpha=1e-5, tau=0.01, discount=0.9, batch_size=12, num_trials=100):
    """
    Simulates the required number of episodes.
        
    PARAMETERS
    ----------
    - data_path(str) : the directory path to the csv data
    - epsilon(float, default=0.9) : the parameter used for choosing greedy actions
    - decay(float, default=1e-3) : the decay parameter for <epsilon> (after every episode)
    - alpha(float, default=1e-3) : the learning rate for the prediction network
    - tau(float, default=1e-3) : the learning rate for the target network
    - discount(float, default=0.9) : the agent's discount factor (used while computing TD error)
    - batch_size(int, default=12) : the batch size for sampling training data
    - num_trials(int, default=10) : the number of replications/trials 
    
    RETURNS
    -------
    - the average profit across trials(scalar)
    - the average loss across trials(scalar)
    - the average win rate across trials(scalar) 
    - the net profits for every episode averaged across trials(array)
    - the net losses for every episode averaged across trials(array)
    - the win rate for every episode averaged across trials(array)
    """
    avg_profit = []
    avg_loss = []
    avg_winrate = []
    trials_profits = [] 
    trials_losses = []
    trials_winrates = []
    for trial in range(num_trials):
        tf.reset_default_graph() # setup a new graph and start all over again
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            Q = Prediction_Network(learning_rate=alpha, discount_rate=discount)
            T = Target_Network(tau=tau)
            trader = Trader(data_path, Q, T, batch_size=batch_size, discount=discount, epsilon=epsilon, decay=decay)
            sess.run(tf.global_variables_initializer())
            T.soft_update(Q, initialize=True)
            mean_episodic_profits, mean_episodic_losses, episodic_winrates = trader.simulate_episodes(sess,num_episodes=100)
            trials_profits.append(mean_episodic_profits)
            trials_losses.append(mean_episodic_losses)
            trials_winrates.append(episodic_winrates)
            avg_profit.append(np.mean(mean_episodic_profits))
            avg_loss.append(np.mean(mean_episodic_losses))
            avg_winrate.append(np.mean(episodic_winrates))
        trader.reset_after_trial()
    return np.mean(avg_profit), np.mean(avg_loss), np.mean(avg_winrate), np.mean(trials_profits, axis=0), \
           np.mean(trials_losses, axis=0), np.mean(trials_winrates, axis=0)
           
if __name__=='__main__':
    sp, sl, swr, p, l, wr = main_run(data_path='../data/NVDA.csv')
    plt.figure(figsize=(13, 7))
    plt.plot(p, color='green', label='Profits per episode(USD)')
    ax = plt.gca()
    ax.grid(color='orange', alpha=0.35)
    ax.set_facecolor('xkcd:black')
    plt.xlabel('Simulated Episode (#)')
    plt.ylabel('Profits in USD')
    plt.title('Learning curve (Profits per episode) : NVDA')
    plt.legend()
    plt.show()
    ## Loss ##
    plt.figure(figsize=(13, 7))
    plt.plot(l, color='red', label='Losses per episode(USD)')
    ax = plt.gca()
    ax.grid(color='orange', alpha=0.35)
    ax.set_facecolor('xkcd:black')
    plt.xlabel('Simulated Episode (#)')
    plt.ylabel('Losses in USD')
    plt.title('Learning curve (Losses per episode) : NVDA')
    plt.legend()
    plt.show()