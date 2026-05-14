#!/usr/bin/env python3
import gymnasium as gym
import statistics

def simple_heuristic(obs):
    angle = obs[4]
    angular_velocity = obs[5]
    vy = obs[3]
    if vy < -0.5:
        return 2
    elif angle > 0.1 or angular_velocity > 0.1:
        return 1
    elif angle < -0.1 or angular_velocity < -0.1:
        return 3
    else:
        return 0

def run_baseline(env, policy_type, episodes=100):
    print(f"\nTEST DE LA BASELINE : {policy_type.upper()} ---")
    all_scores = []
    all_steps = []
    causes_stats = {"safe-landing": 0, "crash": 0, "out-of-view": 0, "sleep": 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        score = 0
        steps = 0
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            elif policy_type == "heuristic":
                action = simple_heuristic(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
        if truncated: 
            causes_stats["sleep"] += 1
        elif terminated:
            if abs(obs[0]) >= 1.0:
                causes_stats["out-of-view"] += 1
            elif reward < -50:
                causes_stats["crash"] += 1
            else:
                causes_stats["safe-landing"] += 1
        all_scores.append(score)
        all_steps.append(steps)

    print(f"Score Moyen       : {statistics.mean(all_scores):.2f} points")
    print(f"⏱Durée Moyenne     : {statistics.mean(all_steps):.2f} steps/épisode")
    print("Raisons de fin d'épisode :")
    for cause, count in causes_stats.items():
        print(f"   - {cause.capitalize():<13}: {count} fois")

def main():
    env = gym.make("LunarLander-v3")
    run_baseline(env, policy_type="random", episodes=100)
    run_baseline(env, policy_type="heuristic", episodes=100)
    env.close()

if __name__ == "__main__":
    main()