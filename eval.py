import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        
# fichier ou on mettra l'evalutation de notre agent, pour l'instant on utilise des actions aléatoires pour tester l'environnement et voir si tout fonctionne correctement.#