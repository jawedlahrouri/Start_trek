#!/usr/bin/env python3
import gymnasium as gym
import torch
import sys
import os
import math
import statistics
import csv
from gymnasium.wrappers import RecordVideo
from src.agents.dqn_agent import DQNAgent, QNetwork
from src.agents.ppo_agent import PPOAgent
from src.agents.doubledqn_agent import DoubleDQNAgent
from src.config_loader import load_config

def calculate_stats(scores):
    """Calcule la moyenne, l'écart type et l'intervalle de confiance à 95%."""
    try:
        if not scores: # retourne des valeurs par défaut si la liste est vide
            return 0.0, 0.0, 0.0
        mean = sum(scores) / len(scores) # calcule le score moyen
        if len(scores) > 1:
            stdev = statistics.stdev(scores) # calcule l'écart type pour voir la dispersion des données
            ci = 1.96 * (stdev / math.sqrt(len(scores))) # calcule l'intervalle de confiance à 95% pour une meilleure précision
        else:
            stdev = 0.0
            ci = 0.0
        return mean, stdev, ci # retourne les statistiques finales calculées
    except Exception as e:
        print(f"Erreur dans calculate_stats: {e}")
        return 0.0, 0.0, 0.0

def get_env(env_config, video_folder=None):
    """Crée et configure l'environnement."""
    try:
        env_name = env_config.get("name", "LunarLander-v3") # définit le nom de l'environnement avec une valeur par défaut
        gravity = env_config.get("gravity")
        enable_wind = env_config.get("enable_wind")
        wind_power = env_config.get("wind_power")
        turbulence_power = env_config.get("turbulence_power")
        
        render_mode = env_config.get("render_mode")
        if video_folder:
            render_mode = "rgb_array"
            
        env_params = {
            "gravity": gravity,
            "enable_wind": enable_wind,
            "wind_power": wind_power,
            "turbulence_power": turbulence_power,
            "render_mode": render_mode
        }
        # Ne conserve que les paramètres qui ne sont pas None
        env_kwargs = {k: v for k, v in env_params.items() if v is not None}
        env = gym.make(env_name, **env_kwargs)

        print("ENVIRONMENT SETTINGS:")
        print(f"- Gravity: {gravity}")
        print(f"- Enable wind: {enable_wind}")
        print(f"- Wind power: {wind_power}")
        print(f"- Turbulence power: {turbulence_power}")
        if video_folder:
            env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, disable_logger=True) # encapsule l'environnement pour enregistrer chaque épisode
        return env # retourne l'environnement prêt à l'emploi
    except Exception as e:
        print(f"Erreur dans get_env: {e}")
        sys.exit(1)

def run_episode(env, agent, algo_name, seed=None):
    """Exécute un seul épisode et retourne le score et la cause de la fin."""
    try:
        obs, info = env.reset(seed=seed) # réinitialise l'environnement à un état frais en utilisant la graine
        done = False
        score = 0
        steps = 0
        
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # transforme les données de l'environnement en données compréhensibles par torch
            with torch.no_grad(): # désactive le calcul des gradients pour économiser de la mémoire et accélérer l'évaluation
                if algo_name == "PPO":
                    action_probs = agent.actor(state_tensor)
                    action = action_probs.argmax().item()
                else:
                    q_values = agent.q_net(state_tensor)
                    action = q_values.argmax().item()
            obs, reward, terminated, truncated, info = env.step(action) # joue l'action sélectionnée dans l'environnement
            score += reward # accumule la récompense pour calculer le score final
            done = terminated or truncated # vérifie si l'épisode est terminé
            steps += 1 # incrémente le compteur d'etape
        cause = "inconnue"
        if truncated: 
            cause = "endormi" # l'agent a mis trop de temps à atterrir
        elif terminated:
            if abs(obs[0]) >= 1.0: 
                cause = "hors-champ" # l'agent a volé en dehors des limites de l'écran
            elif reward < -50: 
                cause = "crash" # l'agent s'est écrasé trop fort sur le sol
            else: 
                cause = "atterrissage-réussi" # l'agent a réussi à atterrir
        return score, cause, steps # retourne les résultats de l'épisode
    except Exception as e:
        print(f"Erreur dans run_episode: {e}")
        return 0, "erreur", 0

def print_hist(data, bins=10):
    """Affiche un histogramme textuel simple de la distribution."""
    try:
        if not data:
            return # ne fait rien s'il n'y a pas de données à tracer
        min_val = min(data) # trouve le score le plus bas
        max_val = max(data) # trouve le score le plus élevé
        if min_val == max_val:
            print(f"  Toutes les valeurs sont exactement {min_val:.2f}")
            return
            
        bin_size = (max_val - min_val) / bins # calcule la taille de chaque bac de l'histogramme
        counts = [0] * bins # initialise un tableau pour compter les éléments dans chaque bac
        for val in data:
            i = int((val - min_val) / bin_size) # trouve à quel bac la valeur appartient
            if i == bins:
                i = bins - 1 # s'assure que la valeur maximale rentre dans le dernier bac
            counts[i] += 1 # incrémente le compteur du bac
            
        print("\nDistribution des scores:")
        max_count = max(counts) # trouve le bac avec le plus d'éléments pour mettre à l'échelle les barres de texte
        for i in range(bins):
            lower = min_val + i * bin_size # calcule la borne inférieure du bac
            upper = lower + bin_size # calcule la borne supérieure du bac
            bar_len = int(counts[i] / max_count * 40) if max_count > 0 else 0 # met à l'échelle la longueur de la barre jusqu'à 40 caractères
            bar = "#" * bar_len # crée la barre visuelle
            print(f"  [{lower:>8.2f} : {upper:>8.2f}] | {counts[i]:>4} | {bar}") # imprime la ligne de l'histogramme
    except Exception as e:
        print(f"Erreur dans print_text_histogram: {e}")

def load_eval_agent(algo_config, env_config, hypers):
    """Charge l'agent configuré pour l'évaluation."""
    try:
        model_path = algo_config.get("save_path") # récupère le chemin où les poids du modèle sont enregistrés
        if not model_path:
            raise ValueError("Chemin des poids du modèle non trouvé dans la configuration ('save_path')")
        if not os.path.exists(model_path):
            fallback_path = os.path.join("assets/agents_save", os.path.basename(model_path)) # essaie de trouver le modèle dans un dossier de secours par défaut
            if os.path.exists(fallback_path):
                model_path = fallback_path # met à jour le chemin s'il est trouvé dans le dossier de secours
                print(f"Chargement du modèle depuis le dossier de secours: {model_path}")
            else:
                raise FileNotFoundError(f"Poids du modèle non trouvés à '{model_path}' ou '{fallback_path}'")
        else:
            print(f"Chargement du modèle: {model_path}")

        env = get_env(env_config) # crée un environnement temporaire pour lire ses dimensions
        state_size = env.observation_space.shape[0] # obtient la taille de l'état d'observation
        action_size = env.action_space.n # obtient le nombre d'actions possibles
        env.close() # détruit l'environnement temporaire
        algo_name = algo_config.get("name") # obtient le nom de l'algorithme

        if algo_name == "DQN":
            agent = DQNAgent(state_size, action_size, hypers) # initialise l'agent DQN
            agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            agent.epsilon = 0.0
        elif algo_name == "DoubleDQN":
            agent = DoubleDQNAgent(state_size, action_size, hypers) # initialise l'agent Double DQN
            agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            agent.epsilon = 0.0
        elif algo_name == "PPO":
            agent = PPOAgent(state_size, action_size, hypers) # initialise l'agent PPO
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        else:
            raise ValueError(f"Algorithme '{algo_name}' non reconnu pour l'évaluation.")
        
        # Mettre les réseaux en mode évaluation (eval)
        if hasattr(agent, "q_net"):
            agent.q_net.eval()
        if hasattr(agent, "actor"):
            agent.actor.eval()
        if hasattr(agent, "critic"):
            agent.critic.eval()
            
        return agent # retourne l'agent entièrement chargé
    except Exception as e:
        print(f"Erreur dans load_eval_agent: {e}")
        sys.exit(1)

def setup_directories(output_dir="eval_output"):
    """Met en place les répertoires de sortie et le fichier CSV."""
    try:
        os.makedirs(output_dir, exist_ok=True) # crée le répertoire de sortie principal s'il n'existe pas
        results_csv = os.path.join(output_dir, "evaluation_results.csv") # définit le chemin pour le CSV des résultats
        video_base_dir = os.path.join(output_dir, "eval_videos") # définit le chemin pour les enregistrements vidéo
        os.makedirs(video_base_dir, exist_ok=True) # crée le répertoire vidéo s'il n'existe pas
        
        with open(results_csv, mode='w', newline='') as f:
            csv_writer = csv.writer(f) # initialise l'écrivain CSV
            csv_writer.writerow(["Seed", "Episode", "Score", "Steps", "Cause"]) # écrit la ligne d'en-tête du CSV
            
        return results_csv, video_base_dir # retourne les chemins générés
    except Exception as e:
        print(f"Erreur dans setup_directories: {e}")
        sys.exit(1)

def evaluate_all_seeds(env_config, agent, algo_name, seeds, results_csv, video_base_dir):
    """Exécute la boucle d'évaluation pour toutes les graines."""
    try:
        total_causes = {"endormi": 0, "hors-champ": 0, "crash": 0, "atterrissage-réussi": 0, "inconnue": 0} # initialise un dictionnaire pour compter les causes de fin
        all_scores = [] # liste pour stocker tous les scores des épisodes
        all_steps = [] # liste pour stocker toutes les longueurs d'épisodes
        seed_means = {} # dictionnaire pour stocker les scores moyens par graine
        eval_seeds = seeds.get("eval", [10, 20, 30, 40, 50]) # obtient la liste des graines à évaluer depuis la configuration
        if isinstance(eval_seeds, int):
            eval_seeds = [eval_seeds] # s'assure que eval_seeds est une liste même si une seule graine est fournie
        episodes_per_seed = seeds.get("episodes_per_seed", 100) # définit combien d'épisodes exécuter par graine
        print(f"Début de l'évaluation sur {len(eval_seeds)} graines ({episodes_per_seed} épisodes par graine)...")
        
        for seed in eval_seeds:
            print(f"\n--- Évaluation de la graine: {seed} ---")
            seed_scores = [] # liste pour stocker les scores de la graine actuelle
            record_video = env_config.get("record_video", True) # par défaut True pour préserver le comportement
            seed_video_dir = os.path.join(video_base_dir, f"seed_{seed}") if record_video else None
            env = get_env(env_config, video_folder=seed_video_dir) # initialise l'environnement pour cette graine spécifique
            
            for ep in range(episodes_per_seed):
                score, cause, steps = run_episode(env, agent, algo_name, seed=seed + ep) # exécute un épisode complet avec une sous-graine unique
                seed_scores.append(score) # sauvegarde le score de l'épisode localement
                all_scores.append(score) # sauvegarde le score de l'épisode globalement
                all_steps.append(steps) # sauvegarde la longueur de l'épisode
                total_causes[cause] += 1 # incrémente le compteur de la cause de fin spécifique
                
                with open(results_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([seed, ep, round(score, 2), steps, cause]) # ajoute les résultats de l'épisode au fichier CSV
                score_avg = sum(seed_scores) / len(seed_scores) # calcule le score moyen glissant pour cette graine
                print(f"Graine {seed} | Ep {ep} | Score: {score:>7.2f} | Moy(graine): {score_avg:>7.2f} | Fin: {cause}")
            s_mean, s_std, s_ci = calculate_stats(seed_scores) # calcule les statistiques finales pour cette graine
            seed_means[seed] = s_mean # stocke le score moyen final de cette graine
            print(f"\nStatistiques pour la graine {seed}:")
            ci_percent = (s_ci / abs(s_mean) * 100) if s_mean != 0 else 0.0
            print(f"  - Score moyen: {s_mean:.2f}")
            print(f"  - Écart type: {s_std:.2f}")
            print(f"  - Int. de conf. 95%: ±{s_ci:.2f} (±{ci_percent:.1f}%)")
            env.close() # ferme proprement l'environnement
        return all_scores, all_steps, total_causes, seed_means # retourne toutes les métriques collectées
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Erreur dans evaluate_all_seeds: {e}")
        sys.exit(1)

def print_final_summary(all_scores, all_steps, total_causes, seed_means, results_csv, video_base_dir):
    """Affiche le résumé final de l'évaluation."""
    try:
        if not all_scores:
            print("Aucune donnée d'évaluation à résumer.") # quitte tôt si aucune donnée n'a été collectée
            return

        final_mean, final_std, final_ci = calculate_stats(all_scores) # calcule les statistiques globales sur toutes les graines
        avg_steps = sum(all_steps) / len(all_steps) # calcule la longueur moyenne globale des épisodes
        success_rate = (total_causes.get("atterrissage-réussi", 0) / len(all_scores)) * 100 # calcule le pourcentage de réussite global
        
        print("\n" + "="*40)
        print("RÉSUMÉ FINAL DE L'ÉVALUATION")
        print(f"Épisodes totaux     : {len(all_scores)}") # affiche le nombre total d'épisodes joués
        grand_ci_percent = (final_ci / abs(final_mean) * 100) if final_mean != 0 else 0.0
        print(f"Moyenne globale     : {final_mean:.2f}")
        print(f"Écart type global   : {final_std:.2f}")
        print(f"Int. de conf. 95%   : ±{final_ci:.2f} (±{grand_ci_percent:.1f}%)")
        print(f"Durée moy. épisode  : {avg_steps:.1f} étapes") # affiche le temps de survie moyen
        print(f"Taux de réussite    : {success_rate:.1f}%") # affiche le taux de réussite d'atterrissage
        print("-" * 20)
        print("Moyennes par graine:")
        for s, s_mean in seed_means.items():
            print(f"  - Graine {s:<4}: {s_mean:.2f}") # imprime le score moyen spécifique pour chaque graine individuelle
        print("-" * 20)
        
        print_hist(all_scores) # imprime la distribution visuelle des scores
        
        print("-" * 20)
        print("Raisons de la fin:")
        for reason, count in total_causes.items():
            percentage = (count / len(all_scores)) * 100 if all_scores else 0 # calcule le pourcentage d'occurrence pour chaque cause
            print(f"  - {reason:12}: {count} ({percentage:.1f}%)") # imprime la distribution des causes
        print("="*40)
        print(f"Logs détaillés enregistrés dans: {results_csv}") # rappelle à l'utilisateur où le CSV est enregistré
        print(f"Vidéos enregistrées dans: {video_base_dir}") # rappelle à l'utilisateur où les vidéos sont enregistrées
    except Exception as e:
        print(f"Erreur dans print_final_summary: {e}")

def main():
    """Point d'entrée principal."""
    try:
        if len(sys.argv) < 2:
            print("Usage: python eval.py <chemin_vers_config.yaml>") # affiche l'usage correct du script si aucune configuration n'est fournie
            sys.exit(1)

        config_path = sys.argv[1] # obtient le chemin de la configuration depuis les arguments de la ligne de commande
        config = load_config(config_path) # parse le fichier de configuration yaml
        if config is None:
            sys.exit(1) # quitte si le fichier de configuration n'a pas pu être chargé

        algo_config = config.get("algorithm", {}) # extrait les paramètres spécifiques à l'algorithme
        env_config = config.get("environment", {}) # extrait les paramètres spécifiques à l'environnement
        hypers = config.get("hyperparameters", {}) # extrait les hyperparamètres du modèle
        seeds = config.get("seeds", {}) # extrait les graines à évaluer

        algo_name = algo_config.get("name")
        agent = load_eval_agent(algo_config, env_config, hypers) # charge l'agent en mémoire
        results_csv, video_base_dir = setup_directories() # prépare les dossiers de sortie

        try:
            all_scores, all_steps, total_causes, seed_means = evaluate_all_seeds( # exécute la boucle d'évaluation réelle
                env_config, agent, algo_name, seeds, results_csv, video_base_dir
            )
            print_final_summary(all_scores, all_steps, total_causes, seed_means, results_csv, video_base_dir) # imprime le rapport complet final
        except KeyboardInterrupt:
            print("\nÉvaluation arrêtée par l'utilisateur.") # gère la sortie si l'utilisateur appuie sur Ctrl+C
    except Exception as e:
        print(f"Erreur dans main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
