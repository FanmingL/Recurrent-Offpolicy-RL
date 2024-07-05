# dmc environments
import gym
import dmc2gym
import numpy as np

# 
class DMCWrapper(gym.Env):
    def __init__(self, domain_name, task_name, seed=None):
        # print(f"domain name: {domain_name}, task name: {task_name}, seed: {seed}")
        self.env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env.unwrapped._step_limit
        self.seed(seed)
        
    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)
