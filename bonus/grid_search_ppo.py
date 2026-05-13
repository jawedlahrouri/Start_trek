#!/usr/bin/env python3
import gymnasium as gym
import itertools
import copy
import json
import os
import csv
import time
import statistics
from datetime import timedelta
import urllib.request
import urllib.error
import concurrent.futures
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agents.ppo_agent import PPOAgent

WEBHOOK_URL = "https://discord.com/api/webhooks/1500467614869094431/jfb5ebL6sMKpXf5BJgSt0VboSH6AQd0BhWM5USgq1YXvurC834SMcw8zQhYLikzikhwk"

def send_discord_message(content):
    if not WEBHOOK_URL: return
    data = json.dumps({"content": content}).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(WEBHOOK_URL, data=data, headers=headers)
    try: urllib.request.urlopen(req)
    except Exception as e: print(f"Erreur Discord: {e}")

def get_end_cause(obs, reward, terminated, truncated):
    if truncated: return "truncated"
    if terminated:
        if abs(obs[0]) >= 1.0: return "out_of_view"
        elif reward < -50: return "crash"
        else: return "landed"
    return "unknown"

def run_single_test(test_id, total_tests, combo, fixed_hypers):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    
    start_time_worker = time.time()
    run_name = f"test_{test_id:03d}_PPO_ep{combo['episodes']}_lr{combo['learning_rate']}_updt{combo['update_timestep']}"
    print(f"Démarrage Test {test_id}/{total_tests} : [{run_name}]")
    
    hypers = copy.deepcopy(fixed_hypers)
    hypers.update(combo)
    hypers["log_dir"] = f"runs/grid_search/PPO/{run_name}"
    episodes_for_this_test = combo["episodes"]
    
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(state_size, action_size, hypers)
    writer = SummaryWriter(log_dir=hypers["log_dir"])
    
    scores = []
    train_causes = {"landed": 0, "crash": 0, "out_of_view": 0, "truncated": 0}
    convergence_ep = -1
    total_time_step = 0  
    
    max_single_score = -9999.0
    ep_max_single_score = -1
    max_avg_score = -9999.0
    ep_max_avg_score = -1
    
    for ep in range(episodes_for_this_test):
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            total_time_step += 1
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, log_prob, reward, value, done)    
        
            if total_time_step % combo["update_timestep"] == 0:
                agent.update(next_obs)
                
            score += reward
            obs = next_obs
            steps += 1
            
            if done:
                cause = get_end_cause(obs, reward, terminated, truncated)
                train_causes[cause] += 1
                
        scores.append(score)
        
        if score > max_single_score:
            max_single_score = score
            ep_max_single_score = ep

        writer.add_scalar("Train/1_Score", score, ep)
        avg_score = sum(scores[-100:]) / min(len(scores[-100:]), 100)
        writer.add_scalar("Train/2_Score_Moyen", avg_score, ep)
        writer.add_scalar("Train/3_Duree_Episode", steps, ep)
        
        if avg_score > max_avg_score:
            max_avg_score = avg_score
            ep_max_avg_score = ep
            
        if convergence_ep == -1 and avg_score >= 200:
            convergence_ep = ep
            
    train_score = sum(scores[-100:]) / min(len(scores[-100:]), 100)

    evaluation_seeds = [10, 42, 67, 677, 999]
    eval_scores_per_seed = {}
    eval_causes = {"landed": 0, "crash": 0, "out_of_view": 0, "truncated": 0}
    eval_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_eval_steps = 0
    
    eval_env = gym.make("LunarLander-v3")
    eval_start_time = time.time()
    
    for seed in evaluation_seeds:
        seed_scores = []
        for _ in range(100): 
            obs, _ = eval_env.reset(seed=seed)
            done = False
            score = 0
            steps = 0
            while not done:
                action, _, _ = agent.select_action(obs) 
                eval_action_counts[action] += 1
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                score += reward
                obs = next_obs
                steps += 1
                if done:
                    cause = get_end_cause(obs, reward, terminated, truncated)
                    eval_causes[cause] += 1
            
            seed_scores.append(score)
            total_eval_steps += steps
            
        avg_for_this_seed = sum(seed_scores) / len(seed_scores)
        eval_scores_per_seed[f"eval_seed_{seed}"] = avg_for_this_seed
        writer.add_scalar(f"Eval/Score_Moyen_Seed_{seed}", avg_for_this_seed, 0)

    eval_total_time = time.time() - eval_start_time

    env.close()
    eval_env.close()
    writer.close()
    eval_list = list(eval_scores_per_seed.values())
    grand_mean_score = sum(eval_list) / len(eval_list)
    max_eval_score = max(eval_list)
    min_eval_score = min(eval_list)
    
    try: eval_std_dev = statistics.stdev(eval_list)
    except Exception: eval_std_dev = 0.0
    
    margin_diff_pct = ((max_eval_score - min_eval_score) / max(abs(max_eval_score), 1.0)) * 100.0
    is_valid = margin_diff_pct <= 5.0
    
    total_train_ep = episodes_for_this_test
    train_pct = {k: (v / total_train_ep) * 100 for k, v in train_causes.items()}
    total_eval_ep = 100 * len(evaluation_seeds)
    eval_pct = {k: (v / total_eval_ep) * 100 for k, v in eval_causes.items()}
    total_actions = sum(eval_action_counts.values())
    action_pct = {k: (v / total_actions) * 100 if total_actions > 0 else 0 for k, v in eval_action_counts.items()}
    
    worker_duration = time.time() - start_time_worker

    result_dict = {
        "test_id": test_id,
        "run_name": run_name,
        "is_valid": is_valid,
        "eval_score_mean": grand_mean_score,
        "eval_std_dev": round(eval_std_dev, 2),
        "margin_diff_pct": margin_diff_pct,
        
        "train_score_final": train_score,
        "train_max_avg_score": round(max_avg_score, 2),
        "train_convergence_ep": convergence_ep if convergence_ep != -1 else "Fail",
        "training_time_sec": round(worker_duration, 2),
        
        "eval_action_0_rien_pct": round(action_pct[0], 2),
        "eval_action_1_gauche_pct": round(action_pct[1], 2),
        "eval_action_2_moteur_pct": round(action_pct[2], 2),
        "eval_action_3_droite_pct": round(action_pct[3], 2),
    }
    
    for key, val in combo.items(): result_dict[f"param_{key}"] = val
    result_dict.update(eval_scores_per_seed)
    
    for k in train_causes.keys():
        result_dict[f"train_{k}_pct"] = round(train_pct[k], 2)
    for k in eval_causes.keys():
        result_dict[f"eval_{k}_pct"] = round(eval_pct[k], 2)
    
    return result_dict

grid_ppo = {
    "algorithm": ["PPO"],
    "episodes": [7500, 10000, 12500, 15000],
    "learning_rate": [3e-4, 5e-4, 3.0e-4],
    "update_timestep": [1024, 2048, 4096],
    "ppo_epochs": [4, 10],
    "epsilon_decay": [0.2, 0.08, 0.35, 0.5],
    "hidden_size": [32, 64, 128]
}

fixed_hypers = {"gamma": 0.99}

if __name__ == "__main__":
    keys_ppo, values_ppo = zip(*grid_ppo.items())
    combinations = [dict(zip(keys_ppo, v)) for v in itertools.product(*values_ppo)]
    total_tests = len(combinations)
    
    leaderboard = []
    
    send_discord_message(f"**DÉMARRAGE DU GRID SEARCH PPO PARALLÉLISÉ**\n"
                         f"**{total_tests} combinaisons** à tester. Décollage imminent ! 🚀\n")
    
    max_workers = 6 
    start_time = time.time()
    
    csv_file_path = "runs/grid_search/PPO/leaderboard_analytics.csv"
    os.makedirs("runs/grid_search/PPO", exist_ok=True)
    csv_headers_written = False
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_test, i+1, total_tests, combo, fixed_hypers): combo for i, combo in enumerate(combinations)}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                test_number = result["test_id"]
                
                if not csv_headers_written:
                    with open(csv_file_path, mode="w", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=result.keys())
                        writer.writeheader()
                        writer.writerow(result)
                    csv_headers_written = True
                else:
                    with open(csv_file_path, mode="a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=result.keys())
                        writer.writerow(result)
                        
                leaderboard.append(result)
                leaderboard = sorted(leaderboard, key=lambda x: x["eval_score_mean"], reverse=True)
                
                tests_completed = len(leaderboard)
                tests_remaining = total_tests - tests_completed
                elapsed_time = time.time() - start_time
                avg_time_per_test = elapsed_time / tests_completed 
                eta_seconds = avg_time_per_test * tests_remaining
                
                eta_formatted = str(timedelta(seconds=int(eta_seconds)))
                elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))
                
                print(f"[{elapsed_formatted}] Test {test_number} terminé. ETA : ~{eta_formatted}")
                is_top_3 = False
                is_top_1 = False
                for rank, entry in enumerate(leaderboard[:3]):
                    if entry["run_name"] == result["run_name"]:
                        is_top_3 = True
                        if rank == 0: is_top_1 = True
                        break
                        
                param_str = "\n".join([f"> `{k}` : {v}" for k, v in result.items() if k.startswith("param_")])

                if is_top_1:
                    msg = (f"🚨 🏆 **NOUVEAU LEADER ABSOLU PPO (TOP 1) !** 🏆 🚨\n"
                           f"**Avancement:** {tests_completed}/{total_tests} (ETA: ⏳ `{eta_formatted}`)\n"
                           f"**Score Éval:** `{result['eval_score_mean']:.2f} pts` | Valid: `{result['is_valid']}`\n"
                           f"**Configuration :**\n{param_str}")
                elif is_top_3:
                    msg = (f"🌟 **NOUVELLE ENTRÉE DANS LE TOP 3 PPO !** 🌟\n"
                           f"**Avancement:** {tests_completed}/{total_tests} (ETA: ⏳ `{eta_formatted}`)\n"
                           f"**Score Éval:** `{result['eval_score_mean']:.2f} pts` | Valid: `{result['is_valid']}`\n"
                           f"**Configuration :**\n{param_str}")

                if is_top_1 or is_top_3:
                    send_discord_message(msg)
                    
            except Exception as e:
                print(f"Erreur critique sur un worker : {e}")

    send_discord_message("@here 🏁 **GRID SEARCH PPO TERMINÉ !** 🏁\nToutes les métriques ont été sauvegardées !")