
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MazeEnv(gym.Env):
    def __init__(self, size=10):
        super(MazeEnv, self).__init__()
        self.size = size
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        self.agent_pos = self.start_pos.copy()
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        
        self.maze = np.zeros((size, size))
        self.maze[2, 2:8] = 1
        self.maze[5, 1:7] = 1
        self.maze[7, 3:9] = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.astype(np.int32), {}

    def step(self, action):
        old_pos = self.agent_pos.copy()
        
        if action == 0: self.agent_pos[0] -= 1
        elif action == 1: self.agent_pos[0] += 1
        elif action == 2: self.agent_pos[1] -= 1
        elif action == 3: self.agent_pos[1] += 1
        
        # Keep in bounds
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.size-1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.size-1)
        
        # Hit wall
        if self.maze[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos
            reward = -5
        else:
            reward = -1
        
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        if terminated: reward = 100
            
        return self.agent_pos.astype(np.int32), reward, terminated, False, {}