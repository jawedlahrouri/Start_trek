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
    try:
        if not scores: # returns default values if the list is empty
            return 0.0, 0.0, 0.0
        mean = sum(scores) / len(scores) # calculates the average score
        if len(scores) > 1:
            stdev = statistics.stdev(scores) # calculates the standard deviation to see data spread
            ci = 1.96 * (stdev / math.sqrt(len(scores))) # calculates the 95% confidence interval for better precision
        else:
            stdev = 0.0
            ci = 0.0
        return mean, stdev, ci # returns the final calculated statistics
    except Exception as e:
        print(f"Error in calculate_stats: {e}")
        return 0.0, 0.0, 0.0

def get_env(env_config, video_folder=None):
    """Creates and configures the environment."""
    try:
        env_name = env_config.get("name", "LunarLander-v3") # sets the environment name with a default fallback
        gravity = env_config.get("gravity", -10.0) # gets the gravity value
        enable_wind = env_config.get("enable_wind", True) # enables or disables wind in the simulation
        wind_power = env_config.get("wind_power", 15.0) # sets the strength of the wind
        turbulence_power = env_config.get("turbulence_power", 1.5) # sets the turbulence intensity
        
        if "render_mode" in env_config:
            render_mode = env_config["render_mode"] # sets how the environment will be rendered
            if video_folder:
                render_mode = "rgb_array" # forces rgb_array rendering if we want to record a video
        else:
            render_mode = None
        
        env = gym.make( # initializes the gymnasium environment with our custom parameters
            env_name,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            render_mode=render_mode
        )
        if video_folder:
            env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, disable_logger=True) # wraps the environment to record every episode
        return env # returns the ready-to-use environment
    except Exception as e:
        print(f"Error in get_env: {e}")
        sys.exit(1)

def run_episode(env, model, seed=None):
    """Runs a single episode and returns the score and cause of termination."""
    try:
        obs, info = env.reset(seed=seed) # resets the environment to a fresh state using the seed
        done = False
        score = 0
        steps = 0
        
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # transforms env data into torch comprehensible data
            with torch.no_grad(): # disables gradient calculation to save memory and speed up evaluation
                q_values = model(state_tensor) # gets the q-values for all possible actions from the model
                action = q_values.argmax().item() # selects the action with the highest q-value
            obs, reward, terminated, truncated, info = env.step(action) # plays the selected action in the environment
            score += reward # accumulates the reward to compute the final score
            done = terminated or truncated # checks if the episode has ended
            steps += 1 # increments the step counter
        cause = "unknown"
        if truncated: 
            cause = "sleep" # the agent took too long to land
        elif terminated:
            if abs(obs[0]) >= 1.0: 
                cause = "out-of-view" # the agent flew outside the screen boundaries
            elif reward < -50: 
                cause = "crash" # the agent crashed too hard on the ground
            else: 
                cause = "safe-landing" # the agent successfully landed
        return score, cause, steps # returns the episode results
    except Exception as e:
        print(f"Error in run_episode: {e}")
        return 0, "error", 0

def print_hist(data, bins=10):
    """Prints a simple text history of the distribution."""
    try:
        if not data:
            return # does nothing if there is no data to plot
        min_val = min(data) # finds the lowest score
        max_val = max(data) # finds the highest score
        if min_val == max_val:
            print(f"  All values are exactly {min_val:.2f}")
            return
            
        bin_size = (max_val - min_val) / bins # calculates the size of each historic bin
        counts = [0] * bins # initializes an array to count items in each bin
        for val in data:
            i = int((val - min_val) / bin_size) # finds which bin the value belongs to
            if i == bins:
                i = bins - 1 # ensures the max value fits in the last bin
            counts[i] += 1 # increments the bin counter
            
        print("\nScore Distribution:")
        max_count = max(counts) # finds the bin with the most items to scale the text bars
        for i in range(bins):
            lower = min_val + i * bin_size # calculates the lower bound of the bin
            upper = lower + bin_size # calculates the upper bound of the bin
            bar_len = int(counts[i] / max_count * 40) if max_count > 0 else 0 # scales the bar length up to 40 characters
            bar = "#" * bar_len # creates the visual bar
            print(f"  [{lower:>8.2f} : {upper:>8.2f}] | {counts[i]:>4} | {bar}") # prints the histogram row
    except Exception as e:
        print(f"Error in print_text_histogram: {e}")

def load_eval_model(algo_config, env_config, hypers):
    """Loads the model based on the configuration."""
    try:
        model_path = algo_config.get("save_path") # retrieves the path where the model weights are saved
        if not model_path:
            raise ValueError("Model weights path not found in config ('save_path')")
        if not os.path.exists(model_path):
            fallback_path = os.path.join("assets/agents_save", os.path.basename(model_path)) # tries to find the model in a default fallback folder
            if os.path.exists(fallback_path):
                model_path = fallback_path # updates the path if found in fallback
                print(f"Loading model from fallback: {model_path}")
            else:
                raise FileNotFoundError(f"Model weights not found at '{model_path}' or '{fallback_path}'")
        else:
            print(f"Loading model: {model_path}")

        env = get_env(env_config) # creates a temporary environment to read its dimensions
        state_size = env.observation_space.shape[0] # gets the size of the observation state
        action_size = env.action_space.n # gets the number of possible actions
        env.close() # destroys the temporary environment
        hidden_size = hypers.get("hidden_size", 64) # retrieves the hidden layer size from the config
        algo_name = algo_config.get("name") # gets the algorithm name

        if "DQN" in algo_name:
            model = QNetwork(state_size, action_size, hidden_size=hidden_size) # initializes DQN network
        elif "DoubleDQN" in algo_name:
            model = DoubleDQNAgent(state_size, action_size, hypers) # initializes Double DQN agent
        elif algo_name == "PPO":
            print("Joseph ajoute ton algo ici") # placeholder for the PPO agent
            sys.exit(1)
        else:
            raise ValueError(f"Algorithm '{algo_name}' not recognized for evaluation.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)) # loads the trained weights into the model structure
        model.eval() # sets the model to evaluation mode (disables dropout, etc.)
        return model # returns the fully loaded model
    except Exception as e:
        print(f"Error in load_eval_model: {e}")
        sys.exit(1)

def setup_directories(output_dir="eval_output"):
    """Sets up the output directories and CSV file."""
    try:
        os.makedirs(output_dir, exist_ok=True) # creates the main output directory if it doesn't exist
        results_csv = os.path.join(output_dir, "evaluation_results.csv") # defines the path for the results CSV
        video_base_dir = os.path.join(output_dir, "eval_videos") # defines the path for the video recordings
        os.makedirs(video_base_dir, exist_ok=True) # creates the video directory if it doesn't exist
        
        with open(results_csv, mode='w', newline='') as f:
            csv_writer = csv.writer(f) # initializes the CSV writer
            csv_writer.writerow(["Seed", "Episode", "Score", "Steps", "Cause"]) # writes the CSV header row
            
        return results_csv, video_base_dir # returns the generated paths
    except Exception as e:
        print(f"Error in setup_directories: {e}")
        sys.exit(1)

def evaluate_all_seeds(env_config, model, seeds, results_csv, video_base_dir):
    """Runs the evaluation loop for all seeds."""
    try:
        total_causes = {"sleep": 0, "out-of-view": 0, "crash": 0, "safe-landing": 0, "unknown": 0} # initializes a dictionary to count termination causes
        all_scores = [] # list to store all episode scores
        all_steps = [] # list to store all episode lengths
        seed_means = {} # dictionary to store average scores per seed
        eval_seeds = seeds.get("eval", [10, 20, 30, 40, 50]) # gets the list of seeds to evaluate from config
        if isinstance(eval_seeds, int):
            eval_seeds = [eval_seeds] # ensures eval_seeds is a list even if only one seed is provided
        episodes_per_seed = 100 # defines how many episodes to run per seed
        print(f"Starting evaluation on {len(eval_seeds)} seeds ({episodes_per_seed} episodes per seed)...")
        
        for seed in eval_seeds:
            print(f"\n--- Evaluating Seed: {seed} ---")
            seed_scores = [] # list to store scores for the current seed
            seed_video_dir = os.path.join(video_base_dir, f"seed_{seed}") # creates a specific video folder for this seed
            env = get_env(env_config, video_folder=seed_video_dir) # initializes the environment for this specific seed
            
            for ep in range(episodes_per_seed):
                score, cause, steps = run_episode(env, model, seed=seed + ep) # runs one full episode with a unique sub-seed
                seed_scores.append(score) # saves the episode score locally
                all_scores.append(score) # saves the episode score globally
                all_steps.append(steps) # saves the episode length
                total_causes[cause] += 1 # increments the specific termination cause counter
                
                with open(results_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([seed, ep, round(score, 2), steps, cause]) # appends the episode results to the CSV file
                score_avg = sum(seed_scores) / len(seed_scores) # calculates the rolling average score for this seed
                print(f"Seed {seed} | Ep {ep} | Score: {score:>7.2f} | Moy(seed): {score_avg:>7.2f} | Fin: {cause}")
            s_mean, s_std, s_ci = calculate_stats(seed_scores) # calculates the final statistics for this seed
            seed_means[seed] = s_mean # stores the final average score of this seed
            print(f"\nStats for Seed {seed}:")
            print(f"  - Average Score: {s_mean:.2f}")
            print(f"  - Std Deviation: {s_std:.2f}")
            print(f"  - 95% Conf. Int: ±{s_ci:.2f}")
            env.close() # cleanly closes the environment
        return all_scores, all_steps, total_causes, seed_means # returns all the collected metrics
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error in evaluate_all_seeds: {e}")
        sys.exit(1)

def print_final_summary(all_scores, all_steps, total_causes, seed_means, results_csv, video_base_dir):
    """Prints the final summary of the evaluation."""
    try:
        if not all_scores:
            print("No evaluation data to summarize.") # exits early if no data was collected
            return

        final_mean, final_std, final_ci = calculate_stats(all_scores) # calculates the global statistics across all seeds
        avg_steps = sum(all_steps) / len(all_steps) # calculates the global average episode length
        success_rate = (total_causes.get("safe-landing", 0) / len(all_scores)) * 100 # calculates the overall success percentage
        
        print("\n" + "="*40)
        print("FINAL EVALUATION SUMMARY")
        print(f"Total episodes      : {len(all_scores)}") # displays the total number of episodes played
        print(f"Overall Average     : {final_mean:.2f}") # displays the overall average score
        print(f"Overall Std Dev     : {final_std:.2f}") # displays the overall data spread
        print(f"Overall 95% CI      : ±{final_ci:.2f}") # displays the global confidence interval
        print(f"Avg Episode Length  : {avg_steps:.1f} steps") # displays the average survival time
        print(f"Success Rate        : {success_rate:.1f}%") # displays the landing success rate
        print("-" * 20)
        print("Means by Seed:")
        for s, s_mean in seed_means.items():
            print(f"  - Seed {s:<4}: {s_mean:.2f}") # prints the specific average score for each individual seed
        print("-" * 20)
        
        print_hist(all_scores) # prints the visual score distribution
        
        print("-" * 20)
        print("Termination Reasons:")
        for reason, count in total_causes.items():
            percentage = (count / len(all_scores)) * 100 if all_scores else 0 # calculates the occurrence percentage for each cause
            print(f"  - {reason:12}: {count} ({percentage:.1f}%)") # prints the cause distribution
        print("="*40)
        print(f"Detailed logs saved in: {results_csv}") # reminds the user where the CSV is saved
        print(f"Videos saved in: {video_base_dir}") # reminds the user where the videos are saved
    except Exception as e:
        print(f"Error in print_final_summary: {e}")

def main():
    """Main entry point."""
    try:
        if len(sys.argv) < 2:
            print("Usage: python eval.py <path_to_config.yaml>") # displays the correct script usage if no config is provided
            sys.exit(1)

        config_path = sys.argv[1] # gets the config path from the command line arguments
        config = load_config(config_path) # parses the yaml configuration file
        if config is None:
            sys.exit(1) # exits if the config file couldn't be loaded

        algo_config = config.get("algorithm", {}) # extracts algorithm specific settings
        env_config = config.get("environment", {}) # extracts environment specific settings
        hypers = config.get("hyperparameters", {}) # extracts model hyperparameters
        seeds = config.get("seeds", {}) # extracts the seeds to evaluate

        model = load_eval_model(algo_config, env_config, hypers) # loads the model into memory
        results_csv, video_base_dir = setup_directories() # prepares output folders

        try:
            all_scores, all_steps, total_causes, seed_means = evaluate_all_seeds( # runs the actual evaluation loop
                env_config, model, seeds, results_csv, video_base_dir
            )
            print_final_summary(all_scores, all_steps, total_causes, seed_means, results_csv, video_base_dir) # prints the final comprehensive report
        except KeyboardInterrupt:
            print("\nEvaluation stopped by user.") # handles exit if the user presses Ctrl+C
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
