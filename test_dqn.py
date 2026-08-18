import torch
import numpy as np
import time
from maze_env import MazeEnv
from train_dqn_proper import DQN # import your DQN class

env = MazeEnv(size=10)
model = DQN(2, 4)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

print("=== TESTING TRAINED DQN AGENT ===")
state, info = env.reset()

for step in range(50):
    x, y = state
    state_tensor = torch.FloatTensor([x, y])
    with torch.no_grad():
        action = model(state_tensor).argmax().item()
    
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    maze_vis = env.maze.copy()
    maze_vis[int(state[0]), int(state[1])] = 8
    print(maze_vis)
    time.sleep(0.3)
    
    if done: 
        print(f"Goal Reached in {step+1} steps!")
        break