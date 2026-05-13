import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, state_size, action_size, hypers):
        self.gamma = hypers.get("gamma", 0.99) # ajustement des valeur douce afin d'eviter les valeurs aberrantes
        self.clip_ratio = hypers.get("clip_ratio", 0.2) # permet de donner le ratio pour le calcul du loss de l'actor network (epsilone)
        self.ppo_epochs = hypers.get("ppo_epochs", 4) # definie le nombre de fois auquel les données stocké dans le batch peuvent etre consulter

        self.actor = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, action_size), nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4) # prepare le bacpropagation
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = []

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