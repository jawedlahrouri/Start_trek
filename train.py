import gymnasium as gym
import torch
import torch.optim as optim
import random
from src.agent import QNetwork, ReplayBuffer, train_step

env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_net = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
memory = ReplayBuffer(capacity=30000)
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
            loss = train_step(q_net, optimizer, states, actions, rewards, next_states, dones)

        score += reward
        obs = next_obs 
    scores.append(score)
    score_avg = sum(scores[-100:]) / len(scores[-100:])
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Épisode {ep} | Score: {score:.2f} | Moyenne (100 ep): {score_avg:.2f} | Epsilon: {epsilon:.2f}")
env.close()
torch.save(q_net.state_dict(), "lunar_lander_model.pth")
print("Modèle sauvegardé avec succès sous 'lunar_lander_model.pth' !")