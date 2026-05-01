#!/usr/bin/env python3
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
import torch.optim as optim
import random
from src.agent import QNetwork, ReplayBuffer, train_step
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/lunar_lander_v2") 

env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_net = QNetwork(state_size, action_size)
target_net = QNetwork(state_size, action_size)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
memory = ReplayBuffer(capacity=30000)

TAU = 1e-3
batch_size = 64
episodes = 800
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
scores = []

for ep in range(episodes):
    obs, info = env.reset()
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
            loss = train_step(q_net, target_net, optimizer, states, actions, rewards, next_states, dones)
            
            episode_losses.append(loss)
            
            for target_param, local_param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        
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
torch.save(q_net.state_dict(), "lunar_lander_model.pth")
print("Modèle sauvegardé avec succès !")