import os
import sys
import random
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from src.config_loader import load_config
from src.agents.dqn_agent import DQNAgent
from src.agents.doubledqn_agent import DoubleDQNAgent
from src.agents.ppo_agent import PPOAgent

def set_global_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
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
    global_seed = seeds_config.get("train", None)
    if isinstance(global_seed, list):
        global_seed = global_seed[0] 
    if global_seed is not None:
        set_global_seed(global_seed)
    env_name = env_config.get("name", "LunarLander-v3")
    record_video = env_config.get("record_video", False)
    raw_render = env_config.get("render_mode")
    if str(raw_render).lower() in ["none", "null"]:
        render_mode = None
    else:
        render_mode = raw_render
    if record_video:
        render_mode = "rgb_array"
    env = gym.make(
        env_name, 
        render_mode=render_mode,
        gravity=env_config.get("gravity", -10.0),
        enable_wind=env_config.get("enable_wind", False),
        wind_power=env_config.get("wind_power", 15.0),
        turbulence_power=env_config.get("turbulence_power", 1.5)
    )
    if record_video:
        video_dir = os.path.join("assets", "videos", algo_config.get("name", "model"))
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: ep % 100 == 0)
        print(f"Enregistrement vidéo activé dans : {video_dir}")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    algo_name = algo_config.get("name")
    if algo_name == "DQN":
        agent = DQNAgent(state_size, action_size, hypers)
    elif algo_name == "DoubleDQN":
        agent = DoubleDQNAgent(state_size, action_size, hypers)
    elif algo_name == "PPO":
        agent = PPOAgent(state_size, action_size, hypers)
    else:
        raise ValueError(f"Algorithme {algo_name} non supporté.")

    writer = SummaryWriter(log_dir=algo_config.get("log_dir"))
    episodes = train_config.get("episodes", 1000)
    scores = []

    print("\nDÉMARRAGE DE L'ENTRAÎNEMENT")

    for ep in range(episodes):
        obs, info = env.reset(seed=global_seed if ep == 0 else None)
        done = False
        score = 0
        steps = 0
        while not done:
            if algo_name == "PPO":
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(obs, action, log_prob, reward, value, done)
            else:
                action = agent.select_action(obs, ep)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.step(obs, action, reward, next_obs, done)
                if hasattr(agent, "update"):
                    agent.update()
            score += reward
            obs = next_obs
            steps += 1
        if algo_name == "PPO":
            agent.update(obs)
        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()
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
        score_avg = sum(scores[-100:]) / min(len(scores[-100:]), 100) 
        logs = agent.get_logs() if hasattr(agent, "get_logs") else {}

        writer.add_scalar("Score/Episode", score, ep)
        writer.add_scalar("Score/Moyenne_100", score_avg, ep)
        writer.add_scalar("Stats/Episode_Length", steps, ep)
        for key, val in logs.items():
            writer.add_scalar(f"Stats/{key.capitalize()}", val, ep)
        writer.add_scalar("Cause/Crash", 1 if cause == "crash" else 0, ep)
        writer.add_scalar("Cause/Safe_landing", 1 if cause == "safe-landing" else 0, ep)

        eps_str = f" | Epsilon: {agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
        print(f"Ep {ep:>4} | Score: {score:>7.2f} | Moy(100): {score_avg:>7.2f} | Fin: {cause:<12}{eps_str}")

    writer.close() 
    env.close()

    save_filename = algo_config.get("save_path", "model.pth")
    save_dir = "assets/agents_save"
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, os.path.basename(save_filename))
    if hasattr(agent, "save"):
        agent.save(final_save_path)
        print(f"\nModèle sauvegardé avec succès dans {final_save_path}")

if __name__ == "__main__":
    main()
