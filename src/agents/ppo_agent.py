import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, state_size, action_size, hypers):
        self.gamma = hypers.get("gamma") 
        self.clip_ratio = hypers.get("epsilon_decay") 
        self.ppo_epochs = hypers.get("ppo_epochs") 
        lr = hypers.get("learning_rate") 
        hidden_size = hypers.get("hidden_size", 64) 

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    def extract_memory_to_tensors(self):
        # Utilisation directe de PyTorch pour créer les tenseurs à partir des listes
        states = torch.tensor([m[0] for m in self.memory], dtype=torch.float32)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.long)
        log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float32)
        rewards = torch.tensor([m[3] for m in self.memory], dtype=torch.float32)
        values = torch.tensor([m[4] for m in self.memory], dtype=torch.float32)
        dones = torch.tensor([m[5] for m in self.memory], dtype=torch.float32)
        
        return states, actions, log_probs, rewards, values, dones

    def select_action(self, state, current_episode=None):
        # transform les donnees de l'env en données comprehensibles pour torch
        state = torch.FloatTensor(state).unsqueeze(0) 

        action_probs = self.actor(state) # recuperation des probas de choix pour les differentes actions
        
        dist = Categorical(action_probs) # vas permettre a l'algo de tenter des chemins nouveaux malgres les pourcentages
        action = dist.sample()
    
        action_log_prob = dist.log_prob(action) # recup la proba exacte de l'action choisie et calcule son log
        state_value = self.critic(state) # definie la critic value pour le state
        return action.item(), action_log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        # stock les valeurs du choix dans la memoire pour les reutiliser plus tard
        self.memory.append((state, action, log_prob, reward, value, done)) 
        
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
        states, actions, old_log_probs, rewards, values, dones = self.extract_memory_to_tensors()
        returns, advantages = self.compute_advantages(rewards, values, dones, next_state) 
        
        for _ in range(self.ppo_epochs):

            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            current_values = self.critic(states).squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # calcule des loss
            actor_loss = -torch.min(surr1, surr2).mean() # mis en neg car on cherche a avoir un gradient ascendant
            critic_loss = nn.MSELoss()(current_values, returns)
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            
            self.optimizer_actor.zero_grad() # vas clean les anciens poids pour laisser place aux nouveaux
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
        checkpoint = torch.load(filename, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

    def get_logs(self):
        return {"actor_loss": self.last_actor_loss, "critic_loss": self.last_critic_loss}
