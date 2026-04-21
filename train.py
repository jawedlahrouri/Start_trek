import gymnasium as gym
import torch
from src.agent import QNetwork

env = gym.make("LunarLander-v3", render_mode="human")
state_list = env.observation_space.shape[0]
action_list = env.action_space.n
q_net = QNetwork(state_list, action_list)
episodes = 500
SEED = 67

for ep in range(episodes):
    obs, info = env.reset(seed=SEED if ep == 0 else None)
    done = False
    score = 0
    
    while not done:
        action = env.action_space.sample() 
        next_obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        obs = next_obs
        done = terminated or truncated
    print(f"Épisode {ep} terminé avec un score de {score:.2f}")
env.close()
