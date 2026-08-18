import numpy as np
import random
from maze_env import MazeEnv

env = MazeEnv(size=10)
q_table = np.zeros((10, 10, 4))

alpha = 0.1   
gamma = 0.99  
epsilon = 1.0 
episodes = 1000

print("Training started with Q-Learning...")

for episode in range(episodes):
    state, _ = env.reset()
    x, y = state
    total_reward = 0
    
    for step in range(100):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[x, y])
        
        next_state, reward, done, _, _ = env.step(action)
        nx, ny = next_state
        
        q_table[x, y, action] += alpha * (reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action])
        
        x, y = nx, ny
        total_reward += reward
        if done: break
    
    epsilon = max(0.01, epsilon * 0.995)
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

print("Training Complete!")
np.save("q_table.npy", q_table)