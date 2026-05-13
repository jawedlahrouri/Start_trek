import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_size, hypers):
        self.gamma = hypers.get("gamma") 
        self.clip_ratio = hypers.get("epsilon_decay") 
        self.ppo_epochs = hypers.get("ppo_epochs") 
        
        lr = hypers.get("learning_rate") 

        self.actor = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, action_size), nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []

    def extract_memory_to_tensors(self):
        # On utilise np.array() pour fusionner les données AVANT de les donner à PyTorch
        states = torch.FloatTensor(np.array([m[0] for m in self.memory]))
        actions = torch.LongTensor(np.array([m[1] for m in self.memory]))
        log_probs = torch.FloatTensor(np.array([m[2] for m in self.memory]))
        rewards = torch.FloatTensor(np.array([m[3] for m in self.memory]))
        values = torch.FloatTensor(np.array([m[4] for m in self.memory]))
        dones = torch.FloatTensor(np.array([m[5] for m in self.memory]))
        
        return states, actions, log_probs, rewards, values, dones
        
        return states, actions, log_probs, rewards, values, dones

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # transforme les donner de l'env en données comprehensible pour torch

        action_probs = self.actor(state) # recuperation des proba de choix pour les differentes actions pour le state de l'actor network
        
        dist = Categorical(action_probs) # vas permettre a l'algo de tenter des chemin nouveau malgres les pourcentages
        action = dist.sample()
    
        action_log_prob = dist.log_prob(action) # recup la proba exact de l'action choisie et calcule sont log pour faciliter les clacules
        state_value = self.critic(state) # definie la critic value pour le state
        return action.item(), action_log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done)) # stock les valeur du choix danhs la memoire pour les reutiliser plus tard (calcule du loss)
        
    def clear_memory(self):
        self.memory = []

    def compute_advantages(self, rewards, values, dones, next_state):
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).item()
            
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1.0 - dones[step])
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_state):
        states, actions, old_log_probs, rewards, values, dones = self.extract_memory_to_tensors() # reprend les valeurs dans le bon type
        returns, advantages = self.compute_advantages(rewards, values, dones, next_state) # calcule la valeur de l'advantage qui vas servir a calculer les loss values
        
        for _ in range(self.ppo_epochs):

            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            current_values = self.critic(states).squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            #calcule des loss
            actor_loss = -torch.min(surr1, surr2).mean() # mis en neg car on cherche a avoir un gradient ascendant afin de garantir la recherche du meilleur score 
            critic_loss = nn.MSELoss()(current_values, returns)
            
            self.optimizer_actor.zero_grad() # vas clean les enciens poids pour laisser place aux nouveau
            actor_loss.backward() # backpropagation sur l'actor network
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        self.clear_memory()

    def save(self, filename):
        # Sauvegarde les deux cerveaux (Acteur et Critique)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filename)

    def load(self, filename):
        # Charge une sauvegarde précédente
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])