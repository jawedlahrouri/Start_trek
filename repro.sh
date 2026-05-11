#!/bin/bash
# Dossier des configurations
CONFIG_DIR="configs"
LOG_DIR="runs"

echo "=== Démarrage de la boucle d'automatisation StarTrek ==="

if ! pgrep -x "tensorboard" > /dev/null
then
    echo "Lancement de Tensorboard..."
    tensorboard --logdir=$LOG_DIR --port=6006 > /dev/null 2>&1 &
    echo "Dashboard disponible sur http://localhost:6006"
else
    echo "Tensorboard est déjà en cours d'exécution."
fi

for config_file in $CONFIG_DIR/*.yaml $CONFIG_DIR/*.yml; do
    [ -e "$config_file" ] || continue
    
    echo "----------------------------------------------------"
    echo "[*] Exécution du run avec la config : $config_file"
    echo "----------------------------------------------------"    
    echo "Entraînement en cours..."
    ./train.py "$config_file"
    if [ $? -eq 0 ]; then
        echo "Entraînement terminé avec succès."
        echo "Évaluation en cours..."
        ./eval.py "$config_file"
    else
        echo "Erreur lors de l'entraînement de $config_file. Passage à la suivante."
    fi
    echo ""
done

echo "=== Tous les runs sont terminés ! ==="
