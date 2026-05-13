# Documentation: How `eval.py` Works

## Why `eval.py`?

The `eval.py` script is designed to evaluate the performance of a trained agent on the LunarLander environment (or any other Gymnasium-compatible environment). Its main purposes are:

- **Load a saved model**: It can load weights for various agents like **DQN**, **DoubleDQN**, **PPO**.
- **Robust Evaluation**: It runs evaluation episodes across multiple seeds to ensure the results are statistically significant and not due to chance.
- **Detailed Statistics**: It generates key performance metrics, including mean score, standard deviation, and the 95% confidence interval.
- **Data Logging**: It saves detailed, episode-by-episode results into a CSV file for further analysis.
- **Visual Analysis**: It records videos of the evaluation episodes, allowing for visual inspection of the agent's behavior.

## How does `eval.py` work?

The script follows a clear, step-by-step process:

1. **Configuration Loading**:

   - The script requires a YAML configuration file as a command-line argument. This file specifies the algorithm, environment settings, hyperparameters, and the seeds to be used for evaluation.
2. **Model Loading**:

   - The trained model is loaded from the path specified in the configuration file. A fallback mechanism is included to search for the model in a default `assets/agents_save/` directory if it's not found at the primary path.
3. **Environment Setup**:

   - The Gymnasium environment is created using the parameters from the config file (e.g., gravity, wind).
   - If video recording is enabled, the environment is wrapped with `RecordVideo` to save each episode.
4. **Evaluation Loop**:

   - The script iterates through each specified seed. By default, it runs 100 episodes per seed.
   - In each episode:
     - The agent acts greedily, always choosing the action with the highest Q-value from the loaded model.
     - Scores, episode termination causes (e.g., `crash`, `safe-landing`, `out-of-view`), and step counts are recorded.
     - The results for the episode are appended to the CSV file.
5. **Statistics and Summary**:

   - After all episodes are completed, the script calculates and displays a final summary.
   - This includes overall and per-seed statistics (mean, std dev, CI).
   - A text-based histogram shows the distribution of scores.
   - A breakdown of termination causes and the overall success rate are printed.
   - Finally, it reminds the user where the CSV results and video files have been saved.

## Key Takeaways

- `eval.py` is the central tool for objectively measuring the performance of your RL agents.
- It ensures **reproducibility** through the use of multiple seeds and systematic data logging.
- It provides a **comprehensive report** (CSV, videos, console stats) to facilitate the analysis and comparison of different models.
- Usage is straightforward: `python eval.py <path_to_config.yaml>`.
- The script is easily extensible to support other algorithms or environments.
