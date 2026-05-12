#!/bin/bash
set -e

CONFIG_DIR="configs"
LOG_DIR="runs"
REQUIREMENTS="requirements.txt"
VENV_DIR=".venv"
DEFAULT_CONFIG="$CONFIG_DIR/default_config.yaml"
LOG_FILE="setup.log"

echo "StarTrek Setup"
echo "Full log in $LOG_FILE"
exec > >(tee "$LOG_FILE") 2>&1

for d in "$CONFIG_DIR" "$LOG_DIR"; do
    [ -d "$d" ] || mkdir -p "$d"
done

if [ -d "$VENV_DIR" ]; then
    echo "Remove old venv: ($VENV_DIR)..."
    rm -rf "$VENV_DIR"
fi
echo "Create python3.12 venv in $VENV_DIR..."
if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV_DIR"
else
    echo "Need python 3.12 on your pc"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "Cleaning previous runs..."
rm -rf "$LOG_DIR"/*

if ! ls "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml 1> /dev/null 2>&1; then
    echo "No config file found, creating $DEFAULT_CONFIG."
    cat > "$DEFAULT_CONFIG" <<EOF
environment:
  name: "LunarLander-v3"

algorithm:
  name: "DQN"
  log_dir: "runs/exp_fast"
  save_path: "exp_fast_model.pth"

hyperparameters:
  learning_rate: 2.0e-3
  gamma: 0.95
  tau: 5.0e-3
  batch_size: 128
  buffer_capacity: 10000
  hidden_size: 32
  epsilon_start: 0.5
  epsilon_decay: 0.9
  epsilon_min: 0.05

seeds:
  train: 123
  eval: 456

training:
  episodes: 10
EOF
fi

if [ -f "$REQUIREMENTS" ]; then
    echo "Checking Python dependencies..."
    pip install -r "$REQUIREMENTS"
else
    echo "[!] $REQUIREMENTS file not found. Skipping dependencies install."
fi

if ! pgrep -x "tensorboard" > /dev/null; then
    echo "Launching TensorBoard..."
    tensorboard --logdir="$LOG_DIR" --port=6006 > /dev/null 2>&1 &
    echo "Dashboard available at http://localhost:6006"
else
    echo "TensorBoard is already running."
fi

SUCCESS=0
FAIL=0
for config_file in "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml; do
    [ -e "$config_file" ] || continue
    echo "----------------------------------------------------"
    echo "[*] Running with config: $config_file"
    echo "----------------------------------------------------"
    echo "Launching training..."
    if python3 train.py "$config_file"; then
        echo "Training launched successfully."
        echo "Launching evaluation..."
        if python3 eval.py "$config_file"; then
            echo "Evaluation launched successfully."
            SUCCESS=$((SUCCESS+1))
        else
            echo "[!] Evaluation failed to launch for $config_file."
            FAIL=$((FAIL+1))
        fi
    else
        echo "[!] Training failed to launch for $config_file. Skipping."
        FAIL=$((FAIL+1))
    fi
    echo ""
done

echo "\n--- ENVIRONMENT ---"
python3 --version
pip freeze
date

echo "\n=== Run summary ==="
echo "Success: $SUCCESS"
echo "Failures: $FAIL"
if [ $FAIL -eq 0 ]; then
    echo "\nAll runs launched successfully!"
else
    echo "\nSome runs failed to launch. See $LOG_FILE for details."
fi
