import os
import gymnasium as gym
import sys
from torch.utils.tensorboard import SummaryWriter
from src.config_loader import load_config
from src.agents.dqn_agent import DQNAgent
from src.agents.doubledqn_agent import DoubleDQNAgent

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "configs/default_config.yaml"
config = load_config(config_path)
env_config = config.get("environment", {})
algo_config = config.get("algorithm", {})
hypers = config.get("hyperparameters", {})
seeds_config = config.get("seeds", {})
train_config = config.get("training", {})
env = gym.make(env_config.get("name"), render_mode=env_config.get("render_mode"))
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

algo_name = algo_config.get("name")
if algo_name == "DQN":
    agent = DQNAgent(state_size, action_size, hypers)
elif algo_name == "DoubleDQN":
    agent = DoubleDQNAgent(state_size, action_size, hypers)
else:
    raise ValueError(f"Algorithme {algo_name} non supporté.")

writer = SummaryWriter(log_dir=algo_config.get("log_dir"))
episodes = train_config.get("episodes")
train_seeds = seeds_config.get("train", [None])
if not isinstance(train_seeds, list):
    train_seeds = [train_seeds]
scores = []

for ep in range(episodes):
    current_seed = train_seeds[ep % len(train_seeds)]
    obs, info = env.reset(seed=current_seed)
    done = False
    score = 0
    steps = 0 
    
    while not done:
        action = agent.select_action(obs, ep)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.step(obs, action, reward, next_obs, done)
        agent.update()
        score += reward
        obs = next_obs
        steps += 1
    if hasattr(agent, "decay_epsilon"):
        agent.decay_epsilon()
        
    cause = "unknown"
    if truncated: cause = "sleep" 
    elif terminated:
        if abs(obs[0]) >= 1.0: cause = "out-of-view"
        elif reward < -50: cause = "crash"
        else: cause = "safe-landing"
    scores.append(score)
    score_avg = sum(scores[-100:]) / min(len(scores[-100:]), 100) 
    logs = agent.get_logs()

    writer.add_scalar("Score/Episode", score, ep)
    writer.add_scalar("Score/Moyenne_100", score_avg, ep)
    writer.add_scalar("Stats/Episode_Length", steps, ep)
    for key, val in logs.items():
        writer.add_scalar(f"Stats/{key.capitalize()}", val, ep)
    writer.add_scalar("Cause/Crash", 1 if cause == "crash" else 0, ep)
    writer.add_scalar("Cause/Safe_landing", 1 if cause == "safe-landing" else 0, ep)
    print(f"Ep {ep} | Seed: {str(current_seed):>4} | Score: {score:>7.2f} | Moy(100): {score_avg:>7.2f} | Fin: {cause}")

writer.close() 
env.close()

save_filename = algo_config.get("save_path", "model.pth")
save_dir = "assets/agents_save"

os.makedirs(save_dir, exist_ok=True)
final_save_path = os.path.join(save_dir, os.path.basename(save_filename))
agent.save(final_save_path)
print(f"Modèle sauvegardé avec succès dans {final_save_path}")