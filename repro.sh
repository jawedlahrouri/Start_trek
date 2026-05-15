#!/bin/bash

# Vérifie si un chemin de fichier de configuration est fourni. Sinon, affiche l'utilisation et quitte.
if [ -z "$1" ]; then
    echo "Erreur : Aucun fichier de configuration fourni."
    echo "Usage: ./repro.sh <chemin_vers_config.yaml>"
    exit 1
fi

# Vérifie si le fichier fourni existe. Sinon, affiche une erreur et quitte.
if [ ! -f "$1" ]; then
    echo "Erreur : Fichier de configuration non trouvé à '$1'"
    exit 1
fi

CONFIG_FILE=$1

# 2. Exécuter l'entraînement
echo " Début de l'entraînement "
echo "Utilisation du fichier de configuration : $CONFIG_FILE"

# Exécute le script d'entraînement avec le fichier de configuration fourni.
python train.py "$CONFIG_FILE"

# Vérifie le code de sortie du script d'entraînement. S'il n'est pas 0, une erreur s'est produite.
if [ $? -ne 0 ]; then
    echo "L'entraînement a échoué. Arrêt du script. "
    exit 1
fi

echo "Entraînement terminé avec succès."
echo ""

# 3. Exécuter l'évaluation
echo "Début de l'évaluation"
echo "Utilisation du même fichier de configuration : $CONFIG_FILE"

# Exécute le script d'évaluation avec le même fichier de configuration.
python eval.py "$CONFIG_FILE"

# Vérifie le code de sortie du script d'évaluation.
if [ $? -ne 0 ]; then
    echo "L'évaluation a échoué."
    exit 1
fi

echo "Évaluation terminée avec succès."
echo ""
echo "Script de reproduction terminé."
