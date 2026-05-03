import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from .base_agent import BaseAgent

class QNetwork(nn.Module):
    def __init__(self, state_list, action_list, hidden_size=64, **kwargs):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_list, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_list)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, hypers):
        super().__init__(state_size, action_size, hypers)
        self.q_net = QNetwork(state_size, action_size, **hypers)
        self.target_net = QNetwork(state_size, action_size, **hypers)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=hypers.get("learning_rate"))
        self.memory = ReplayBuffer(capacity=hypers.get("buffer_capacity"))
        self.batch_size = hypers.get("batch_size")
        self.gamma = hypers.get("gamma")
        self.tau = hypers.get("tau")
        self.epsilon = hypers.get("epsilon_start")
        self.epsilon_decay = hypers.get("epsilon_decay")
        self.epsilon_min = hypers.get("epsilon_min")
        self.last_loss = 0.0

    def select_action(self, state, current_ep):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, target_q_values)
        self.last_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_logs(self):
        return {"loss": self.last_loss, "epsilon": self.epsilon}

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)