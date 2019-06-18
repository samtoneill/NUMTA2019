''' 
Class implementations of the following agents.

* Exponential Weights with Semi-bandit Feedback
* Exponential Weights with Bandit Feedback
* e-Greedy with Semi-bandit Feedback
* e-Greedy with Bandit Feedback

'''

import numpy as np
from itertools import islice

class EWSemiAgent:
  def __init__(self, env, actions):
    self.env = env
    self.actions = actions
    self.uniform = np.ones(len(self.actions))/len(self.actions)
    self.P = []
    self.W = [np.ones(len(self.actions)) ]
    self.gamma = 0.01
    self.epsilon = 0
    self.rewards = []
    self.choices = []
    
  # choose action based on probability distribution self.P
  def choose_action(self, epsilon=None):
    
    if epsilon is None:
      epsilon = self.epsilon
      
    # update probability distribution with e-hedge 
    self.P.append(epsilon*self.uniform + (1-epsilon)*(self.W[-1]/np.sum(self.W[-1])))
    action = int(self.env.np_random.choice(range(len(self.actions)), 1, p=self.P[-1]))
    self.choices.append(self.actions[action])
    return self.actions[action]
  
  # get the corresponding reward from the environment
  def update_reward(self):
    self.rewards.append(self.env.reward(self.choices[-1]))
    return np.float(self.rewards[-1])
  
  # update the agent  weights 
  def update_weights(self, gamma=None):
    if gamma is None:
      gamma = self.gamma
    # payoff vector
    est_pay_off_v = []
    
    for action in self.actions:
      est_pay_off = 0
      # calculate the path costs for each chosen action (path)
      for u,v in zip(action[:-1], action[1:]):
        # get corresponding bandit cost
        est_pay_off += self.env.G[u][v]['weight']
      
      # append to payoff vector, note we scale function (unbounded)
      est_pay_off_v.append((-1)*est_pay_off/1000)
    # convert to np array
    est_pay_off_v = np.array(est_pay_off_v)
    # apply update to weights
    W = np.array(self.W[-1])
    W *= np.exp(gamma*est_pay_off_v/len(self.actions))
    self.W.append(W)

class EWFullAgent:
  
  def __init__(self, env, actions):
    self.env = env
    self.actions = actions
    self.uniform = np.ones(len(self.actions))/len(self.actions)
    self.P = []
    self.W = [np.ones(len(self.actions)) ]
    self.gamma = 0.01
    self.epsilon = 0
    self.rewards = []
    self.choices = []
    
  def choose_action(self, epsilon=None):
    if epsilon is None:
      epsilon = self.epsilon
    self.P.append(epsilon*self.uniform + (1-epsilon)*(self.W[-1]/np.sum(self.W[-1])))
    choice = int(self.env.np_random.choice(range(len(self.actions)), 1, p=self.P[-1]))
    self.choices.append(choice)
    return self.actions[choice]
  
  def update_reward(self):
    action = self.actions[self.choices[-1]]
    self.rewards.append(self.env.reward(action))
    return np.float(self.rewards[-1])
  
  def update_weights(self, gamma=None):
    if gamma is None:
      gamma = self.gamma
    choice = self.choices[-1]
    action = self.actions[choice]
    est_pay_off = 0
    # calculate the path costs for each chosen action (path)
    for u,v in zip(action[:-1], action[1:]):
      # pull the corresponding bandit
      est_pay_off += (-1)*self.env.G[u][v]['weight']/(self.P[-1][choice]*1000)
      
    W = np.array(self.W[-1])
    W[choice] *= np.exp(gamma*est_pay_off/len(self.actions))
    self.W.append(W)

class EGreedySemiAgent:
  
  def __init__(self, env, actions,epsilon=0.1):
    self.env = env
    self.actions = actions
    self.uniform = np.ones(len(self.actions))/len(self.actions)
    self.W = [np.zeros(len(self.actions)) ]
    self.epsilon = epsilon
    self.rewards = []
    self.choices = []
    self.t = 0
    
  def choose_action(self,epsilon=None):
    if epsilon is None:
      epsilon = self.epsilon
    if self.env.np_random.random_sample() < epsilon:
      choice = int(self.env.np_random.choice(range(len(self.actions)), 1, p=self.uniform))
    else:
      choice = np.argmax(self.W[-1])
    self.choices.append(choice)
    return self.actions[choice]
  
  def update_reward(self):
    action = self.actions[self.choices[-1]]
    self.rewards.append(self.env.reward(action))
    return np.float(self.rewards[-1])
  
  # update the agent  weights 
  def update_weights(self):
    # payoff vector
    self.t += 1
    est_pay_off_v = []
    
    for action in self.actions:
      est_pay_off = 0
      # calculate the path costs for each chosen action (path)
      for u,v in zip(action[:-1], action[1:]):
        # get corresponding bandit cost
        est_pay_off += self.env.G[u][v]['weight']
      
      # append to payoff vector, note we scale function (unbounded)
      est_pay_off_v.append((-1)*est_pay_off/1000)
    # convert to np array
    est_pay_off_v = np.array(est_pay_off_v)
    # apply update to weights
    W = np.array(self.W[-1])
    W += (1/(self.t+1))*(est_pay_off_v - W)
    self.W.append(W)

class EGreedyFullAgent:
  def __init__(self, env, actions,epsilon=0.1):
    self.env = env
    self.actions = actions
    self.uniform = np.ones(len(self.actions))/len(self.actions)
    self.W = [np.zeros(len(self.actions)) ]
    self.epsilon = epsilon
    self.rewards = []
    self.choices = []
    self.t = 0
    
  def choose_action(self, epsilon=None):
    if epsilon is None:
      epsilon = self.epsilon
    if self.env.np_random.random_sample() < epsilon:
      choice = int(self.env.np_random.choice(range(len(self.actions)), 1, p=self.uniform))
    else:
      choice = np.argmax(self.W[-1])
    self.choices.append(choice)
    return self.actions[choice]
  
  def update_reward(self):
    action = self.actions[self.choices[-1]]
    self.rewards.append(self.env.reward(action))
    return np.float(self.rewards[-1])
  
  # update the agent  weights 
  def update_weights(self):
    # payoff vector
    self.t += 1
    choice = self.choices[-1]
    action = self.actions[choice]
    est_pay_off = 0
    # calculate the path costs for each chosen action (path)
    for u,v in zip(action[:-1], action[1:]):
      # pull the corresponding bandit
      est_pay_off += (-1)*self.env.G[u][v]['weight']/1000
      
    
    W = np.array(self.W[-1])
    W[choice] += (1/self.t+1)*(est_pay_off - W[choice])
    
    
    self.W.append(W) 

def algorithm_loop(GBanditGame, ods, gammas=None, epsilons=None):
  od_rewards = []
  edge_info = []
  while True:
    round_actions = []
    for agents in ods:
      for agent in agents:
        if epsilons is None:
          GBanditGame.play(agent.choose_action())
        else:
          GBanditGame.play(agent.choose_action(epsilon=epsilons[GBanditGame.t]))

    edge_info.append([(d['id'],np.float(d['weight']),np.float(d['bandit'].no_pulls)) for (u,v,d) in sorted(GBanditGame.G.edges(data=True))])

    rewards = []
    for agents in ods:
      for agent in agents:
        reward = agent.update_reward()

        rewards.append(reward)
    od_rewards.append(rewards)
    
    for agents in ods:
      for agent in agents:
        if gammas is None:
          agent.update_weights()
        else:
          agent.update_weights(gamma=gammas[GBanditGame.t])

    done = GBanditGame.step()
    if done:
      break
      
    GBanditGame.reset_bandits()
  return od_rewards, edge_info
  