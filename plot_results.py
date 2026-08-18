import numpy as np
import matplotlib.pyplot as plt

ql_rewards = np.load("qlearning_rewards.npy")
dqn_rewards = np.load("dqn_rewards.npy")

plt.figure(figsize=(10,5))
plt.plot(ql_rewards, label="Q-Learning")
plt.plot(dqn_rewards, label="DQN")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning vs DQN on 10x10 Maze")
plt.legend()
plt.grid()
plt.savefig("training_plot.png", dpi=300)
print("Plot saved as training_plot.png")