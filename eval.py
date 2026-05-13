#!/usr/bin/env python3
import gymnasium as gym
import torch
import sys
import os
import math
import statistics
import csv
from gymnasium.wrappers import RecordVideo
from src.agents.dqn_agent import QNetwork
from src.agents.doubledqn_agent import DoubleDQNAgent
from src.config_loader import load_config

def calculate_stats(scores):
    """Calculates mean, standard deviation, and 95% confidence interval."""
    if not scores:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        stdev = statistics.stdev(scores)
        ci = 1.96 * (stdev / math.sqrt(len(scores)))
    else:
        stdev = 0.0
        ci = 0.0
    return mean, stdev, ci

def get_env(env_config, video_folder=None):
    """Creates and configures the environment."""
    env_name = env_config.get("name", "LunarLander-v3")
    gravity = env_config.get("gravity", -10.0)
    enable_wind = env_config.get("enable_wind", True)
    wind_power = env_config.get("wind_power", 15.0)
    turbulence_power = env_config.get("turbulence_power", 1.5)
    
    if "render_mode" in env_config:
        render_mode = env_config["render_mode"]
        if video_folder:
            render_mode = "rgb_array"
    else:
        render_mode = None
    
    env = gym.make(
        env_name,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode=render_mode
    )
    
    if video_folder:
        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, disable_logger=True)
        
    return env

def run_episode(env, model, seed=None):
    """Runs a single episode and returns the score and cause of termination."""
    obs, info = env.reset(seed=seed)
    done = False
    score = 0
    steps = 0
    
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
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
            
    return score, cause, steps

def print_text_histogram(data, bins=10):
    """Prints a simple text-based histogram of the distribution."""
    if not data:
        return
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        print(f"  All values are exactly {min_val:.2f}")
        return
        
    bin_size = (max_val - min_val) / bins
    counts = [0] * bins
    for val in data:
        i = int((val - min_val) / bin_size)
        if i == bins:
            i = bins - 1
        counts[i] += 1
        
    print("\nScore Distribution:")
    max_count = max(counts)
    for i in range(bins):
        lower = min_val + i * bin_size
        upper = lower + bin_size
        bar_len = int(counts[i] / max_count * 40) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{lower:>8.2f} : {upper:>8.2f}] | {counts[i]:>4} | {bar}")

def main():
    """Parse arguments and run evaluation"""
    if len(sys.argv) < 2:
        print("Usage: python eval.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    if config is None:
        sys.exit(1)

    algo_config = config.get("algorithm", {})
    env_config = config.get("environment", {})
    hypers = config.get("hyperparameters", {})
    seeds = config.get("seeds", {})

    model_path = algo_config.get("save_path")
    if not model_path:
        print(f"Error: Model weights path not found in config ('save_path')")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        fallback_path = os.path.join("assets/agents_save", os.path.basename(model_path))
        if os.path.exists(fallback_path):
            model_path = fallback_path
            print(f"Loading model from fallback: {model_path}")
        else:
            print(f"Error: Model weights not found at '{model_path}' or '{fallback_path}'")
            sys.exit(1)
    else:
        print(f"Loading model: {model_path}")

    env = get_env(env_config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_size = hypers.get("hidden_size", 64)
    algo_name = algo_config.get("name")

    if "DQN" in algo_name:
        model = QNetwork(state_size, action_size, hidden_size=hidden_size)
    elif "DoubleDQN" in algo_name:
        model = DoubleDQNAgent(state_size, action_size, hypers)
    elif algo_name == "PPO":
        print("Joseph ajoute ton algo ici")
        sys.exit(1)
    else:
        print(f"Error: Algorithm '{algo_name}' not recognized for evaluation.")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    output_dir = "eval_output"
    os.makedirs(output_dir, exist_ok=True)
    
    results_csv = os.path.join(output_dir, "evaluation_results.csv")
    video_base_dir = os.path.join(output_dir, "eval_videos")
    os.makedirs(video_base_dir, exist_ok=True)
    
    with open(results_csv, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Seed", "Episode", "Score", "Steps", "Cause"])

    total_causes = {"sleep": 0, "out-of-view": 0, "crash": 0, "safe-landing": 0, "unknown": 0}
    all_scores = []
    all_steps = []
    seed_means = {}

    eval_seeds = seeds.get("eval", [10, 20, 30, 40, 50])
    if isinstance(eval_seeds, int):
        eval_seeds = [eval_seeds]
    
    episodes_per_seed = 100
    print(f"Starting evaluation on {len(eval_seeds)} seeds ({episodes_per_seed} episodes per seed)...")
    
    try:
        for seed in eval_seeds:
            print(f"\n--- Evaluating Seed: {seed} ---")
            seed_scores = []
            
            seed_video_dir = os.path.join(video_base_dir, f"seed_{seed}")
            env = get_env(env_config, video_folder=seed_video_dir)
            
            for ep in range(episodes_per_seed):
                score, cause, steps = run_episode(env, model, seed=seed + ep)
                
                seed_scores.append(score)
                all_scores.append(score)
                all_steps.append(steps)
                total_causes[cause] += 1
                
                with open(results_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([seed, ep, round(score, 2), steps, cause])
                
                score_avg = sum(seed_scores) / len(seed_scores)
                print(f"Seed {seed} | Ep {ep} | Score: {score:>7.2f} | Moy(seed): {score_avg:>7.2f} | Fin: {cause}")

            s_mean, s_std, s_ci = calculate_stats(seed_scores)
            seed_means[seed] = s_mean
            print(f"\nStats for Seed {seed}:")
            print(f"  - Average Score: {s_mean:.2f}")
            print(f"  - Std Deviation: {s_std:.2f}")
            print(f"  - 95% Conf. Int: ±{s_ci:.2f}")
            
            env.close()

        if all_scores:
            final_mean, final_std, final_ci = calculate_stats(all_scores)
            avg_steps = sum(all_steps) / len(all_steps)
            success_rate = (total_causes["safe-landing"] / len(all_scores)) * 100
            
            print("\n" + "="*40)
            print("FINAL EVALUATION SUMMARY")
            print(f"Total episodes      : {len(all_scores)}")
            print(f"Overall Average     : {final_mean:.2f}")
            print(f"Overall Std Dev     : {final_std:.2f}")
            print(f"Overall 95% CI      : ±{final_ci:.2f}")
            print(f"Avg Episode Length  : {avg_steps:.1f} steps")
            print(f"Success Rate        : {success_rate:.1f}%")
            print("-" * 20)
            print("Means by Seed:")
            for s, s_mean in seed_means.items():
                print(f"  - Seed {s:<4}: {s_mean:.2f}")
            print("-" * 20)
            
            print_text_histogram(all_scores)
            
            print("-" * 20)
            print("Termination Reasons:")
            for reason, count in total_causes.items():
                print(f"  - {reason:12}: {count} ({count/len(all_scores)*100:.1f}%)")
            print("="*40)
            print(f"Detailed logs saved in: {results_csv}")
            print(f"Videos saved in: {video_base_dir}")

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

if __name__ == "__main__":
    main()
