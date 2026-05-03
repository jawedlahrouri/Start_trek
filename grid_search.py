#!/usr/bin/env python3
import gymnasium as gym
import itertools
import copy
import json
import urllib.request
import urllib.error
from src.agents.dqn_agent import DQNAgent

WEBHOOK_URL = "https://discord.com/api/webhooks/1500467614869094431/jfb5ebL6sMKpXf5BJgSt0VboSH6AQd0BhWM5USgq1YXvurC834SMcw8zQhYLikzikhwk"

def send_discord_message(content):
    if not WEBHOOK_URL:
        return
    
    data = json.dumps({"content": content}).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    req = urllib.request.Request(WEBHOOK_URL, data=data, headers=headers)
    try:
        urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        print(f"Erreur Discord: {e}")

grid_soft = {
    "algorithm": ["DQN"],
    "episodes": [400, 500, 800, 1000],
    "update_type": ["soft"],
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [64, 128],
    "tau": [1e-3, 5e-3],
    "target_update_freq": [None],
    "epsilon_decay": [0.995, 0.99],
    "buffer_capacity": [10000, 30000, 50000],
    "hidden_size": [64, 128]
}

grid_hard = {
    "algorithm": ["DQN"],
    "episodes": [400, 500, 800, 1000],
    "update_type": ["hard"],
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [64, 128],
    "tau": [None],
    "target_update_freq": [500, 1000],
    "epsilon_decay": [0.995, 0.99],
    "buffer_capacity": [10000, 30000, 50000],
    "hidden_size": [64, 128]
}

fixed_hypers = {"gamma": 0.99, "epsilon_start": 1.0, "epsilon_min": 0.01}
keys_soft, values_soft = zip(*grid_soft.items())
combos_soft = [dict(zip(keys_soft, v)) for v in itertools.product(*values_soft)]
keys_hard, values_hard = zip(*grid_hard.items())
combos_hard = [dict(zip(keys_hard, v)) for v in itertools.product(*values_hard)]
combinations = combos_soft + combos_hard
total_tests = len(combinations)

leaderboard = []
send_discord_message(f"**DÉMARRAGE DU GRID SEARCH SUR LE VPS**\n"
                     f"**{total_tests} combinaisons** à tester. On se revoit dans quelques jours !")

for i, combo in enumerate(combinations):
    test_number = i + 1
    print(f"Test {test_number}/{total_tests} en cours...")
    hypers = copy.deepcopy(fixed_hypers)
    hypers.update(combo)
    episodes_for_this_test = combo["episodes"]
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, hypers)
    scores = []
    for ep in range(episodes_for_this_test):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(obs, ep)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(obs, action, reward, next_obs, done)
            agent.update()
            score += reward
            obs = next_obs
        agent.decay_epsilon()
        scores.append(score)
    env.close()
    
    final_avg_score = sum(scores[-100:]) / min(len(scores[-100:]), 100)
    leaderboard.append({"score": final_avg_score, "params": combo})
    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    is_top_3 = False
    is_top_1 = False
    for rank, entry in enumerate(leaderboard[:3]):
        if entry["params"] == combo:
            is_top_3 = True
            if rank == 0:
                is_top_1 = True
            break
    param_str = "\n".join([f"> `{k}` : {v}" for k, v in combo.items()])

    if is_top_1:
        msg = (f"🚨 🏆 **NOUVEAU LEADER ABSOLU (TOP 1) !** 🏆 🚨\n"
               f"**Avancement:** {test_number}/{total_tests}\n"
               f"**Score Moyen:** `{final_avg_score:.2f} pts` 🚀\n"
               f"**Configuration :**\n{param_str}")
    elif is_top_3:
        msg = (f"🌟 **NOUVELLE ENTRÉE DANS LE TOP 3 !** 🌟\n"
               f"**Avancement:** {test_number}/{total_tests}\n"
               f"**Score Moyen:** `{final_avg_score:.2f} pts`\n"
               f"**Configuration :**\n{param_str}")
    else:
        msg = (f"⚙️ Test {test_number}/{total_tests} terminé.\n"
               f"**Algo:** `{combo['algorithm']}` | **Score:** `{final_avg_score:.2f} pts`\n"
               f"(Pas de changement au sommet).")

    send_discord_message(msg)
    print(f"Fin du test {test_number}. Score: {final_avg_score:.2f}")

podium_msg = "@here 🏁 **LE GRID SEARCH EST TOTALEMENT TERMINÉ !** 🏁\nVoici les 3 meilleures configurations trouvées :\n\n"

for rank, entry in enumerate(leaderboard[:3]):
    medals = ["🥇", "🥈", "🥉"]
    podium_msg += f"{medals[rank]} **Place #{rank+1}** - Score: `{entry['score']:.2f} pts`\n"
    podium_msg += f"```json\n{json.dumps(entry['params'], indent=2)}\n```\n"

send_discord_message(podium_msg)
print("Entraînement massif terminé. Check Discord !")
