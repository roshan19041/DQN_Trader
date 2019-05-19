#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:01:06 2019

@author: roshanprakash
"""

import math
import numpy as np
import pandas as pd

class Trader:
    
    """ Agent class """
    
    def __init__(self, data_path, QNet, TNet, batch_size=10, discount=0.9, epsilon=0.5, decay=1e-3):
        """ 
        Initializes the Trader (the agent).
        
         PARAMETERS
        ----------
        - data_path(str) : the directory path to the OHLCV data used to extract state representations \
                      and train/test the agent
        - Q_Net(Q_Network object) : the prediction network used to approximate Q_values for every action in any state
        - TNet(Target_Network object) : the target network used to approximate target Q_values every action \
                                              in any state
        - batch_size(int, default=10) : the training batch size
        - discount(float, default=0.9) : the discount factor for the agent
        - epsilon(float, default=0.5) : the parameter for the epsilon-greedy action choice
        - decay(float, default=1e-7) : the decay factor for epsilon

        RETURNS
        -------
        - None
        """    
        # Simulation/ Environment specific variable initializations 
        data = self.get_data(data_path)
        split_idx = math.floor(0.65*data[0].shape[0])
        self.data_train = data[0][:split_idx]
        self.data_test = data[0][split_idx:]
        self.state_train = data[1][:split_idx]
        self.state_test = data[1][split_idx:]
        self.l = 5 # the number of sequential instances used to extract a state representation
        self.prediction_network = QNet
        self.target_network = TNet
        self.batch_size = batch_size
        self.memory_limit = 180
        
        # Agent specific variable initializations (episodic) 
        self.inventory = 0 
        self.portfolio_value = self.margin_balance = 2500000.0 
        self.bought_prices = [] 
        self.profits = []
        self.losses = []
        self.current_state = None
        self.previous_action = 0 
        self.actions = [1, 0, -1] # [buy, neutral, sell]
        self.trade_size = 1000 
        self.discount = 0.99
        self.transaction_term = 0.015*self.trade_size 
        self.buys = 0
        self.max_buys = 25
        
        # agent's replay memory
        self.actions_idxs_taken = [] 
        self.states_visited = [] 
        self.rewards = [] 
        
        # Algorithm specific parameters
        self.epsilon = self.epsilon_start = epsilon
        self.eps_decay = decay
        
    def compute_MA(self, series, long_term=True):
        """
        Computes long-term/short-term Moving Averages for the entries in 
        the data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - long_term (bool, default=True) : If True, uses a 
          longer lag time (200 days)
        
        RETURNS
        -------
        - A pandas series containing the MA's for every timestamp (row index).
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        if long_term:
            lag = 200
        else:
            lag = 50
        assert len(temp)>lag, 'Not enough data points in this timeseries!'
        for idx in range(lag, len(temp)):
            temp[idx] = series[idx-lag:idx].mean()
        temp[:lag] = None
        return temp
        
    def compute_EMA(self, series, num_days=50):
        """
        Computes Exponential Moving Averages of prices for every timestamp
        in the data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - num_days (int, default=20) : the smoothing period
        
        RETURNS
        -------
        - A pandas series containing EMA's for every timestamp (row index).
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        smoothing_factor = 2/(num_days+1)
        EMA_prev = 0.0
        for idx in range(len(temp)):
            EMA_current = (temp[idx]*smoothing_factor)+EMA_prev*(1-smoothing_factor)
            # update values for next iteration
            temp[idx] = EMA_current
            EMA_prev = EMA_current 
        return temp
    
    def compute_momentum_signals(self, series):
        """
        Computes Moving Average Convergence Divergence and signal line
        for entries in data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        
        RETURNS
        -------
        - Two pandas series containing MACD and it's associated signal line 
          for every timestamp (row index).   
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        t1 = self.compute_EMA(temp, num_days=12)
        t2 = self.compute_EMA(temp, num_days=26)
        MACD = t1-t2
        signal_line = self.compute_EMA(MACD, num_days=9)
        return MACD, signal_line
    
    def get_data(self, data_path): 
        """ 
        Reads the historical quotes data.
        
        PARAMETERS
        ----------
        - data_path(str) : the directory path to the OHLCV data used to extract state representations \
                           and train/test the agent
        RETURNS
        -------
        - a numpy array with indices corresponding (in order) to daily_close, daily_trade_volume, \
          daily_open, daily_high, daily_low, day, month, year
        """
        data = pd.read_csv(data_path, usecols=['Open', 'Close', 'High', 'Low', 'Volume'])
        state_data = data.copy()
        state_data['MA'] = self.compute_MA(state_data['Close'], long_term=False)
        state_data['EMA'] = self.compute_EMA(state_data['Close'])
        state_data['delta'] = state_data['Close']-state_data['MA']
        state_data['MACD'], state_data['Signal'] = self.compute_momentum_signals(state_data['Close'])
        col_filter = ['Open', 'Close', 'High', 'Low', 'Volume','MA', 'EMA']
        state_data[col_filter] = state_data[col_filter].pct_change()
        state_data.dropna(inplace=True)
        # for consistency in correspondence between state data and main data,
        # skip the first (50+1) indices in main data for lag=50 MA and percent change computations respectively
        data = data[51:].values
        state_data = state_data.values
        # normalize the state data
        state_data = (state_data-np.amin(state_data, axis=0))/(np.amax(state_data, axis=0)-np.amin(state_data, axis=0))
        return [data, state_data]
    
    def get_state(self, data_idx, action, is_training=True): 
        """ 
        Returns the state representation extracted from the data's features.
        
        PARAMETERS
        ----------
        - data_idx : the data index of the instance for which the state representation needs to be extracted
        - action : the action used to encode a one-hot-vector into the state representation
        - is_training(bool, default=True) : if True, training mode

        RETURNS
        -------
        - the state representation ; a list.
        """
        position_vector = [0, 0, 0]
        position_idx = self.actions.index(action)
        position_vector[position_idx] = 1
        if is_training:
            state = self.state_train[data_idx-(self.l-1):data_idx+1].flatten()
        else:
            state = self.state_test[data_idx-(self.l-1):data_idx+1].flatten()
        state = np.append(state, position_vector)
        return np.expand_dims(state, axis=0)
     
    def epsilon_greedy_choice(self, greedy_choice):
        """ 
        Chooses action according to the epsilon-greedy policy.
        
        PARAMETERS
        ----------  
        - greedy_choice : the greedy action chosen by the agent, based on Q-values
        
        RETURNS
        -------
        - an action according to the epsilon-greedy policy ; (1, 0 or -1).
        """
        action_choices = np.arange(3)
        action_choice_probs = [self.epsilon/3]*3
        action_choice_probs[greedy_choice]+=(1-self.epsilon)
        action = self.actions[np.random.choice(action_choices, 1, p=action_choice_probs)[0]]
        return action
            
    def check_action_feasibility(self, action, opening_price, inventory, margin_balance): 
        """ 
        Checks the feasibility of an intended action.
        
        PARAMETERS
        ----------  
        - action : the intended action ---> should be -1, 0 or 1 corresponding to sell, neutral and buy respectively.
        - opening_price : the opening price of the stock for the day the intended trade is planned to be executed.
        - inventory(int) : the agent's position
        - margin_balance(float) : the agent's account balance
        
        RETURNS
        -------
        - a bool, indicating the feasibilty of the intended action.
        """
        # neutral position ; always possible 
        if action==0: 
            return True    
        # sell/close position ; check inventory
        elif action==-1: 
            if inventory>=self.trade_size:
                return True 
            else:
                return False      
        # buy/open position ; check cash holdings  
        else: 
            if margin_balance>=self.trade_size*opening_price+self.transaction_term*opening_price and self.buys<self.max_buys:
                return True   
            else:
                return False
            
    def compute_portfolio_value(self, current_balance, inventory, closing_price): 
        """ 
        Computes the current portfolio value of the trader (agent) by accessing it's inventory and monetary holdings.
        
        PARAMETERS
        ----------  
        - current_balance(float) : the current account balance in USD 
        - inventory(int) : the agent's inventory of stocks on that day 
        - closing_price(float) : a float indicating the day's closing price for the day
        
        RETURNS
        -------
        - a float indicating the agent's new portfolio value.
        """
        try: 
            return current_balance+inventory*closing_price 
        except: 
            raise ValueError('Missing arguments. Try again!')       
            
    def compute_reward(self, v_new, v_previous, action):
        """ 
        The reward function that computes the agent's reward for any state transition.
        
        PARAMETERS
        ----------
        - v_new(float) : the portfolio value after the agent's transition to the new state
        - v_previous(float) : the portfolio value before the agent's transition to the new state
        - action(int) : the action taken by the agent in the state (-1, 0 or -1)
            
        RETURNS
        -------
        - a float indicating the agent's reward (for the specified state transition).
        """
        if action==0:
            reward = min(0.0, v_new/v_previous)  
        else: 
            reward = v_new/v_previous        
        return reward
    
    def compute_profits(self, selling_price):
        """ 
        Computes the daily profit from the day's close position.
        Assumptions : the agent will sell a fixed volume of shares for the current day's opening price.
                      the agent will dispose the stock that was bought earliest i.e., the earliest inventory will be sold off.
                      the buying price for that day when the earliest inventory was stocked up/bought will be considered in computing the profit.
        
        PARAMETERS
        ----------
        - selling_price : the current day's opening price ; should be a float
        
        RETURNS
        -------
        - a float indicating the daily return from a sell/close position.
        """
        return (selling_price-self.bought_prices[0])*self.trade_size-(self.transaction_term*selling_price)
      
    def check_stop_loss(self, current_price):
        """ 
        Checks the stop-loss condition.
        
        PARAMETERS
        ----------
        - current_price(float) : current day's opening price
            
        RETURNS
        -------
        - a list of indices of stocks in the agent's purchase history that fail the stop-loss condition.
        """
        prices = []
        for buy_price in self.bought_prices[:-1]:
            if current_price<0.90*buy_price: # threshold of 90% / maximum accrued loss = 10% of purchase price per unit of stock
                prices.append(buy_price)
        return prices
        
    def reset_after_episode(self):
        """ Resets the agent's system characteristics for next episode."""
        self.inventory = 0
        self.buys = 0
        self.portfolio_value = self.margin_balance = 2500000.0
        self.bought_prices = []
        self.current_state = None
        self.previous_action = 0
        self.profits = []
        self.losses = []
        
    def reset_after_trial(self):
        """ Resets the agent's system characteristics for next trial."""
        self.actions_idxs_taken = []
        self.states_visited = []
        self.rewards = []
        self.epsilon = self.epsilon_start
     
    def train_step(self, sess):
        """ 
        Implements a training routine by sampling a fixed-length (k) sequence of states from the agent's memory.
        
        PARAMETERS
        ----------
        - sess : a tensorflow session
        
        RETURNS
        -------
        - None
        """
        assert len(self.states_visited)-1>=self.batch_size, 'Insufficient experience memory for the agent!'
        sampled_idxs = np.random.randint(low=0, high=len(self.states_visited)-1, size=self.batch_size)
        temp = np.array(self.states_visited)
        states_in = temp[sampled_idxs]
        action_idxs = np.expand_dims(np.array(self.actions_idxs_taken)[sampled_idxs], axis=1)
        rewards = np.expand_dims(np.array(self.rewards)[sampled_idxs], axis=1)
        target_states = temp[sampled_idxs+1]
        Q_ = sess.run(self.target_network.model_out, feed_dict={self.target_network.x:target_states})
        Q_targets = np.max(Q_, axis=1, keepdims=True)
        sess.run(self.prediction_network.train_op, feed_dict={self.prediction_network.x:states_in, \
                        self.prediction_network.a:action_idxs, self.prediction_network.r:rewards, \
                        self.prediction_network.Q_t:Q_targets})
        self.target_network.soft_update(self.prediction_network) # soft update the target network's weights
    
    def simulate_episodes(self, sess, is_training=True, num_episodes=100): 
        """ 
        Simulates the required number of episodes.
        
        PARAMETERS
        ----------
        - sess : a tensorflow session
        - is_training(bool, default=True) : indicates training time or test time simulation
        - num_episodes (int, default=100) : the number of episodes to simulate    
        
        RETURNS
        -------
        - a list containing the net daily return(profit/loss) for every episode
        """
        open_idx = 0
        close_idx = 3
        # check to access training data or test time data
        if is_training:
            data = self.data_train
            num_episodes = num_episodes
        else:
            data = self.data_test
            num_episodes = 1
        net_profits = [] # the history of net (episodic) profits
        net_losses = [] # the history of net (episodic) losses
        win_rates = [] # the history of episodic win rates (profitable_transactions/num_transactions)
        for episode in range(num_episodes):
            # initialize episodic counters
            timestep = 1
            num_transactions = 0 # number of transactions
            profitable_transactions = 0 # number of profitable transactions
            # sample a starting state and initialize a risk-reward record holder
            data_idx = np.random.randint(low=self.l, high=len(data)-self.memory_limit)
            self.current_state = self.get_state(data_idx, self.previous_action, is_training)
            while timestep<=self.memory_limit:
                next_open = data[data_idx+1, open_idx]
                next_close = data[data_idx+1, close_idx]
                Q_s = sess.run(self.prediction_network.model_out, feed_dict={self.prediction_network.x:self.current_state})
                greedy_action = self.epsilon_greedy_choice(np.argmax(Q_s)) 
                # check action feasibility
                if self.check_action_feasibility(greedy_action, next_open, self.inventory, self.margin_balance):
                    current_action = greedy_action
                else:
                    current_action = 0 # choose neutral position since intended action is not feasible
                # update buy records if needed
                if current_action==1:
                    self.bought_prices.append(next_open)
                    self.buys+=1
                    num_transactions+=1
                elif current_action==-1: # 'sell'
                    PoL = self.compute_profits(next_open)
                    num_transactions+=1
                    if PoL>=0:
                        profitable_transactions+=1
                        self.profits.append(PoL)
                    else:
                        self.losses.append(PoL)
                    self.bought_prices.pop(0)
                    self.buys-=1
                # save experience to memory
                self.states_visited.append(self.current_state[0])
                self.actions_idxs_taken.append(self.actions.index(current_action))
                # now, update agent's system characteristics
                self.current_state = self.get_state(data_idx+1, current_action, is_training)
                self.margin_balance+=(-current_action*self.trade_size*next_open)-(self.transaction_term*\
                                         next_open*abs(current_action))
                self.inventory+=current_action*self.trade_size
                # check stop-loss and dispose losing investments
                l_prices = self.check_stop_loss(next_open)
                for buy_price in l_prices:
                    self.losses.append(next_open-buy_price)
                    num_transactions+=1
                    self.bought_prices.remove(buy_price)
                    self.buys-=1
                    # update system characteristics
                    self.margin_balance+=(self.trade_size-self.transaction_term)*next_open
                    self.inventory-=self.trade_size
                v_new = self.compute_portfolio_value(self.margin_balance, self.inventory, next_close)
                self.rewards.append(self.compute_reward(v_new, self.portfolio_value, current_action))
                self.portfolio_value = v_new
                # implement training routine, if required
                if timestep>self.batch_size and timestep%self.batch_size==0 and is_training:
                    self.train_step(sess)  
                timestep+=1
                data_idx+=1
            net_profits.append(np.sum(self.profits))
            net_losses.append(np.sum(self.losses))
            win_rates.append(profitable_transactions/num_transactions)
            self.reset_after_episode()
            self.epsilon*=np.exp(-self.eps_decay)
        return net_profits, net_losses, win_rates  