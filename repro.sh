#!/bin/bash

# Check if a configuration file path is provided. If not, print usage and exit.
if [ -z "$1" ]; then
    echo "Error: No configuration file provided."
    echo "Usage: ./repro.sh <path_to_config.yaml>"
    exit 1
fi

# Check if the provided file exists. If not, print an error and exit.
if [ ! -f "$1" ]; then
    echo "Error: Configuration file not found at '$1'"
    exit 1
fi

CONFIG_FILE=$1

# 2. Run Training
echo " Starting Training "
echo "Using configuration file: $CONFIG_FILE"

# Execute the training script with the provided config file.
python train.py "$CONFIG_FILE"

# Check the exit code of the training script. If it's not 0, an error occurred.
if [ $? -ne 0 ]; then
    echo "Training failed. Aborting script. "
    exit 1
fi

echo "Training completed successfully."
echo ""

# 3. Run Evaluation
echo "Starting Evaluation"
echo "Using the same configuration file: $CONFIG_FILE"

# Execute the evaluation script with the same config file.
python eval.py "$CONFIG_FILE"

# Check the exit code of the evaluation script.
if [ $? -ne 0 ]; then
    echo "Evaluation failed."
    exit 1
fi

echo "Evaluation completed successfully."
echo ""
echo "Reproduction script finished."
