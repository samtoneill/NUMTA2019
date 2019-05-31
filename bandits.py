import numpy as np

''' Class for individual Functional Bandit'''

class FunctionalBandit():
  
  def __init__(self, f, noise_scale =1):
    # initialise the no of pulls to 0
    self.no_pulls = 0
    self.noise_scale = noise_scale
    self.f = f
    
  def __str__(self):
    return 'Functional bandit with function {} and scale={}'.format(self.f,self.noise_scale)
  
  def sample(self, x=None):
    if x is None:
      # cast to avoid referring to updated no_pulls below
      self.no_pulls += 1    
      x = int(self.no_pulls)
      
    return self.f(x) + np.random.normal(0, self.noise_scale)
    
  def reset(self):
    self.no_pulls = 0
    return self.no_pulls

class BPRBandit(FunctionalBandit):
  def __init__(self, a, b, c, n, noise_scale =1):
    self.a = a
    self.b = b
    self.c = c
    self.n = n
    f = lambda x: a + b*(x/c)**n
    self.f_int = lambda x: a*x + (b/((n+1)*c**n))*x**(n+1) 
    super(BPRBandit, self).__init__(f=f, noise_scale=noise_scale)
    
  def __str__(self):
    return '{} + {}*(x/{})^{}'.format(self.a, self.b, self.c, self.n)

''' Class for Graph BPR Bandit Game'''


class GraphBPRBanditGame():
  def __init__(self, G, T = 1):

    # note the importance of seeding!!!
    self.G = G
    self.T = T
    self.t = 0
    self.state = None
    
    
  def create_bandits(self):
    
    self.G.graph['a'] = self.np_random.randint(low=6, high=10, size=len(self.G.edges()))
    self.G.graph['b'] = self.np_random.randint(low=6, high=10, size=len(self.G.edges()))
    self.G.graph['c'] = self.np_random.randint(low=1, high=6, size=len(self.G.edges()))
    self.G.graph['n'] = self.np_random.randint(low=1, high=3, size=len(self.G.edges()))
    
    a = self.G.graph['a']
    b = self.G.graph['b']
    c = self.G.graph['c']
    n = self.G.graph['n']
    for index, (u,v,d) in enumerate(sorted(self.G.edges(data=True))):
      d['a'] = a[index]
      d['b'] = b[index]
      d['c'] = c[index]
      d['n'] = n[index]
      d['bandit'] = BPRBandit(a[index],b[index],c[index],n[index], noise_scale=1)
      d['weight'] = d['bandit'].sample(x=0)
    
    
  def seed(self, seed=None):
    self.np_random = np.random.RandomState(seed=seed)
    return seed


  def play(self, action):
       
    for u,v in zip(action[:-1], action[1:]):
      # pull the corresponding bandit
      self.G[u][v]['weight'] = self.G[u][v]['bandit'].sample()


  
  def reward(self, action):
    reward = 0
    # calculate the path costs for each chosen action (path)
    for u,v in zip(action[:-1], action[1:]):
      # pull the corresponding bandit
      reward += self.G[u][v]['weight']
      
    return reward
  
  def step(self):
    self.t += 1
    done = self.t >= self.T
    # reward is not valid here as all agents must pull before observing rewards
    self.reset_bandits()
    return done
  
  
  def __str__(self):
    return '{} bandits and max trials = {}'.format(self.no_bandits, self.t)

  def reset(self, bandits=None):
    self.state = None
    if bandits is None:
      self.bandits = self.create_bandits()  
    else:
      self.bandits = bandits
    self.t = 0
    return self.state
  
  def reset_bandits(self):
    for u,v in self.G.edges():
      self.G[u][v]['bandit'].reset()
      
  def current_bandit_pulls(self):
    return [self.G[u][v]['bandit'].no_pulls for u,v in self.G.edges()]