import numpy as np
import gymnasium as gym
from maze_env import MazeEnv

env = MazeEnv(size=10)

EPISODES = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
MAX_STEPS = 200

q_table = np.zeros((env.size, env.size, env.action_space.n))
rewards_log = []

print("TRAINING STARTED")

for episode in range(EPISODES):
    try:
        state, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            steps += 1
            x, y = int(state[0]), int(state[1])
            
            if np.random.random() < EPSILON: 
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[x, y])

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            nx, ny = int(state[0]), int(state[1])
            
            q_table[x, y, action] = q_table[x, y, action] + ALPHA * (reward + GAMMA * np.max(q_table[nx, ny]) - q_table[x, y, action])
            total_reward += reward

        EPSILON = max(0.01, EPSILON * 0.995)
        rewards_log.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")

    except Exception as e:
        print("CRASH AT EPISODE", episode)
        print(e)
        break

print("Training Complete!")
np.save("q_table.npy", q_table)
np.save("qlearning_rewards.npy", rewards_log)