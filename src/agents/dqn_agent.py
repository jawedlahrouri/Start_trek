import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from src.agents.base_agent import BaseAgent

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states])
        action_batch = torch.tensor(actions, dtype=torch.int64)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        next_state_batch = torch.stack([torch.as_tensor(ns, dtype=torch.float32) for ns in next_states])
        done_batch = torch.tensor(dones, dtype=torch.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, hypers):
        super().__init__(state_size, action_size, hypers)
        self.batch_size = hypers.get("batch_size")
        self.gamma = hypers.get("gamma")
        self.update_type = hypers.get("update_type", "soft")
        self.tau = hypers.get("tau", 1e-3)
        self.target_update_freq = hypers.get("target_update_freq", 1000)
        self.steps_done = 0
        self.epsilon = hypers.get("epsilon_start")
        self.epsilon_min = hypers.get("epsilon_min")
        self.epsilon_decay = hypers.get("epsilon_decay")
        hidden_size = hypers.get("hidden_size", 64)
        self.q_net = QNetwork(state_size, action_size, hidden_size)
        self.target_net = QNetwork(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        lr = hypers.get("learning_rate")
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        buffer_capacity = hypers.get("buffer_capacity", 10000)
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, current_episode=None):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def update(self):
        if len(self.memory) < self.batch_size:
            return   
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (self.gamma * max_next_q_values * (1 - dones))
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.update_type == "soft":
            for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)      
        elif self.update_type == "hard":
            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_logs(self):
        return {"loss": self.last_loss, "epsilon": self.epsilon}

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
