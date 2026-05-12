#!/usr/bin/env python3
import gymnasium as gym
import torch
import sys
import os
from src.agents.dqn_agent import QNetwork
from src.config_loader import load_config

GRAVITY = -10.0
WIND_POWER = 15.0
TURBULENT_POWER = 1.5

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    config = load_config(config_path)
    if config is None:
        print(f"Error: Could not load config from {config_path}")
        sys.exit(1)

    algo_config = config.get("algorithm", {})
    env_config = config.get("environment", {})
    hypers = config.get("hyperparameters", {})
    seeds = config.get("seeds", {})

    model_path = algo_config.get("save_path")
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path} (check 'save_path' in config)")
        sys.exit(1)

    print(f"Using config: {config_path}")
    print(f"Loading model: {model_path}")

    env_name = env_config.get("name", "LunarLander-v3")
    env = gym.make(
        env_name,
        gravity=GRAVITY,
        enable_wind=True,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENT_POWER
    )
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_size = hypers.get("hidden_size", 64)

    model = QNetwork(state_size, action_size, hidden_size=hidden_size)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        print(f"Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    obs, info = env.reset(seed=seeds.get("eval", 10))
    print("Starting evaluation... (Ctrl+C to quit)")
    
    total_reward = 0
    grand_total_reward = 0
    episode_count = 0

    try:
        while True:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                episode_count += 1
                grand_total_reward += total_reward
                print(f"Episode {episode_count} finished | Score: {total_reward:.2f}")
                
                if episode_count >= 100:
                    print("\nEvaluation completed.")
                    print(f"Average score on 100 episodes: {grand_total_reward / episode_count:.2f}")
                    break
                
                total_reward = 0
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
