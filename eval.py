#!/usr/bin/env python3
import gymnasium as gym
import sys
from src.config_loader import load_config

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "configs/default_config.yaml"

config = load_config(config_path)
env_config = config.get("environment", {})
seeds = config.get("seeds", {})

env = gym.make(env_config.get("name"), render_mode=env_config.get("render_mode"))
obs, info = env.reset(seed=seeds.get("eval"))

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        
# fichier ou on mettra l'evalutation de notre agent, pour l'instant on utilise des actions aléatoires pour tester l'environnement et voir si tout fonctionne correctement.#