# DQN_Trader

DQN_Trader is a Reinforcement Learning Agent that is trained to make daily trading decisions to maximize long-run profits while managing a basic portfolio with only NVDA stocks. The length of trading activity is 180 days and the learning procedure must be periodically updated to match the market performance of the stock. This is an experimental attempt to quantify the performance of an AI-based agent in daily trading. Free historical data for stocks that are traded on the NYSE or elsewhere, with sufficient granularity, are not available for public access. Rather than synthesizing artificial data to mimick a stock's timeseries to allow for a more realistic online learning, the current approach to devising this AI-based trading strategy, which focused more on understanding whether an AI-agent can be trained to make optimal trading choices, an offline learning procedure was followed by sampling (180-day) sequences of available data from the last three years for every episode (with possible overlap across episodes). The trading model's logic could easily be extended to work in a more realistic setting.

The agent uses a simple feed-forward, fully-connected neural network to choose an action from the action space, {buy, sell, hold}. While training, the weights of the network are adjusted according the Q-Learning algorithm to minimize TD-error. The learning curve for the agent shows convergence to near-optimal profits per episode. 

Python Dependencies:
-------------------
- tensorflow
- 

