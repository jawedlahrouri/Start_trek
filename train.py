import gymnasium as gym
import torch
from src.agent import QNetwork
from src.config_loader import load_config

config = load_config()
env_config = config.get("environment", {})
algo_config = config.get("algorithm", {})
hypers = config.get("hyperparameters", {})
seeds = config.get("seeds", {})
train_config = config.get("training", {})

env = gym.make(env_config.get("name"), render_mode=env_config.get("render_mode"))
state_list = env.observation_space.shape[0]
action_list = env.action_space.n

q_net = QNetwork(state_list, action_list, **hypers)

episodes = train_config.get("episodes")
SEED = seeds.get("train")

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
