# Start Trek : LunarLander-v3 Reinforcement Learning

Bienvenue dans le projet **Start Trek**. L'objectif de ce projet est de concevoir, d'entraîner et d'évaluer un agent d'Intelligence Artificielle capable d'atterrir en toute sécurité sur la Lune dans l'environnement `LunarLander-v3` de Gymnasium, en utilisant des algorithmes d'Apprentissage par Renforcement (Reinforcement Learning)
---

## Installation & Setup

```bash
# 1. Cloner le dépôt
git clone git@github.com:EpitechPGE2-2025/G-AIA-401-LIL-4-1-starttrek-4.git
cd start_trek

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

L'architecture du projet a été pensée pour séparer l'exécution de la logique métier. Tous les scripts exécutables se trouvent à la racine.

### 1. Évaluation
Pour lancer l'évaluation de notre meilleur modèle sur 5 seeds différentes (100 épisodes par seed) et calculer l'écart-type ainsi que l'intervalle de confiance à 95% :

```bash
python eval.py <fichier_de_configs>
```
*Note : Ce script charge automatiquement notre modèle situé dans `assets/agents_save/`. Les vidéos des atterrissages seront sauvegardées dans `eval_output/eval_videos/`.*

### 2. Entraînement
Pour lancer un nouvel entraînement basé sur un fichier de configuration YAML :

```bash
python train.py <fichier_de_configs>
```
*Le script utilise par défaut les paramètres définis dans `configs/default_configs.yaml`. Vous pouvez suivre l'apprentissage en direct avec TensorBoard via la commande : `tensorboard --logdir runs/`.*

### 3. Recherche d'Hyperparamètres
Pour relancer notre infrastructure de Grid Search massive :

```bash
python grid_search.py
```

### 4. Baseline
Pour visualiser les performances d'un agent agissant de manière totalement aléatoire (heuristique de base) :

```bash
python baseline.py
```

## Architecture du Projet

Le code a été refondu en Programmation Orientée Objet (POO) pour permettre l'expérimentation fluide de différents algorithmes (DQN, DoubleDQN, PPO) via de simples fichiers de configuration.

```text
.
├── baseline.py                 # Heuristique aléatoire
├── eval.py                     # Script d'évaluation officiel (95% CI)
├── grid_search.py              # Script d'ablation distribué
├── train.py                    # Pipeline d'entraînement principal
├── configs/                    # Fichiers de configuration (.yaml)
│   ├── default_dqn.yaml
│   └── elite_dqn.yaml          # Nos meilleurs hyperparamètres
├── src/                        # Logique métier et architecture POO
│   ├── agents/
│   │   ├── base_agent.py       # Interface commune
│   │   ├── dqn_agent.py
│   │   ├── doubledqn_agent.py
│   │   └── ppo_agent.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
├── assets/                     # Modèles pré-entraînés
│   └── agents_save/
│       └── best_dqn_model.pth  # Modèle validant l'Acceptance Criteria
├── documentation/              # Preuves scientifiques et historiques
│   ├── Documentation_Globale.md
│   └── dqn/
└── notebooks/                  # Analyse des données du Grid Search
    └── annexes.ipynb
```

## Documentation

Ce projet contient une documentation pour expliquer chacun de nos choix. Nous avons mené une démarche analytique poussée pour justifier chaque hyperparamètre. 

Nous vous invitons fortement à consulter le dossier `documentation/` pour :

- Avoir un compte rendu global du projet (`Rapport_start_trek.pdf`)
- Comprendre l'évolution de notre pipeline d'entraînement (`train.md`).
- Lire les justifications mathématiques de l'architecture de nos agents (`dqn.md` / `Documentation PPO.pdf`).
- Consulter l'analyse de nos résultats de Grid Search prouvant la prévention de l'overfitting.
- Des graphiques de corrélation et des analyses de distribution sont également disponibles et documentés dans `notebooks/annexes.ipynb`.

## Auteurs
- Joseph Birck
- Lucas Lhomme
- Jawed Lahrouri
