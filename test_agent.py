import numpy as np
import time
from maze_env import MazeEnv

env = MazeEnv(size=10)
q_table = np.load("q_table.npy") # Load the trained brain

print("=== TESTING TRAINED AGENT ===")
print("Running 3 times to prove it learned\n")

for run in range(3):
    state, info = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    print(f"--- RUN {run+1} ---")
    
    while not done and steps < 100:
        steps += 1
        x, y = int(state[0]), int(state[1])
        
        # Pick the BEST action from Q-table. No randomness
        action = np.argmax(q_table[x, y])

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Show the maze
        maze_vis = env.maze.copy()
        maze_vis[x,y] = 8 # Mark agent as 8
        maze_vis[9,9] = 2 # Goal
        print(maze_vis)
        time.sleep(0.2) # Pause so you can see it move
    
    print(f"Goal Reached in {steps} steps! Total Reward: {total_reward}\n")
    print("-------------------\n")

print("TEST COMPLETE")