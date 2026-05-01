#!/usr/bin/env python3
import gymnasium as gym
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
import sys
from src.agent import QNetwork, ReplayBuffer, train_step
from src.config_loader import load_config

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "configs/default_config.yaml"

config = load_config(config_path)
env_config = config.get("environment", {})
algo_config = config.get("algorithm", {})
hypers = config.get("hyperparameters", {})
seeds = config.get("seeds", {})
train_config = config.get("training", {})

# Setup environment
env = gym.make(env_config.get("name"), render_mode=env_config.get("render_mode"))
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Setup networks
q_net = QNetwork(state_size, action_size, **hypers)
target_net = QNetwork(state_size, action_size, **hypers)
target_net.load_state_dict(q_net.state_dict())

# Setup optimizer and memory
optimizer = optim.Adam(q_net.parameters(), lr=hypers.get("learning_rate"))
memory = ReplayBuffer(capacity=hypers.get("buffer_capacity"))

# Tensorboard
writer = SummaryWriter(log_dir=algo_config.get("log_dir"))

# Training parameters
episodes = train_config.get("episodes")
batch_size = hypers.get("batch_size")
gamma = hypers.get("gamma")
tau = hypers.get("tau")
epsilon = hypers.get("epsilon_start")
epsilon_decay = hypers.get("epsilon_decay")
epsilon_min = hypers.get("epsilon_min")
seed = seeds.get("train")

scores = []

for ep in range(episodes):
    obs, info = env.reset(seed=seed if ep == 0 else None)
    done = False
    score = 0
    steps = 0 
    episode_losses = [] 
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.argmax().item() 
                
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        memory.push(obs, action, reward, next_obs, done)
        
        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            loss = train_step(q_net, target_net, optimizer, states, actions, rewards, next_states, dones, gamma=gamma)
            
            episode_losses.append(loss)
            
            for target_param, local_param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        score += reward
        obs = next_obs
        steps += 1
        
    cause = "unknown"
    if truncated:
        cause = "sleep" 
    elif terminated:
        if abs(obs[0]) >= 1.0: 
            cause = "out-of-view"
        elif reward < -50:
            cause = "crash"
        else:
            cause = "safe-landing"
    scores.append(score)
    score_avg = sum(scores[-100:]) / len(scores[-100:])
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
    writer.add_scalar("Score/Episode", score, ep)
    writer.add_scalar("Score/Moyenne_100", score_avg, ep)
    writer.add_scalar("Stats/Epsilon", epsilon, ep)
    writer.add_scalar("Stats/Loss", avg_loss, ep)
    writer.add_scalar("Stats/Episode_Length", steps, ep)
    writer.add_scalar("Cause/Crash", 1 if cause == "crash" else 0, ep)
    writer.add_scalar("Cause/Out_of_view", 1 if cause == "out-of-view" else 0, ep)
    writer.add_scalar("Cause/Sleep", 1 if cause == "sleep" else 0, ep)
    writer.add_scalar("Cause/Safe_landing", 1 if cause == "safe-landing" else 0, ep)
    print(f"Ep {ep} | Score: {score:>7.2f} | Moy(100): {score_avg:>7.2f} | Eps: {epsilon:.2f} | Fin: {cause}")

writer.close() 
env.close()

save_path = algo_config.get("save_path")
torch.save(q_net.state_dict(), save_path)
print(f"Modèle sauvegardé avec succès dans {save_path} !")