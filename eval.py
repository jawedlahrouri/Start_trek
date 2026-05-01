import gymnasium as gym
from src.config_loader import load_config

config = load_config()
env_config = config.get("environment", {})

env = gym.make(env_config.get("name"), render_mode=env_config.get("render_mode"))
obs, info = env.reset(seed=env_config.get("seed"))

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        
# fichier ou on mettra l'evalutation de notre agent, pour l'instant on utilise des actions aléatoires pour tester l'environnement et voir si tout fonctionne correctement.