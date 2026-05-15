# Documentation DQN : Projet Start Trek

## 1. Objectif du projet
L'objectif de ce projet est de développer un agent d'apprentissage par renforcement capable d'atterrir en toute sécurité dans l'environnement LunarLander-v3. Le critère d'acceptation principal exige un score moyen supérieur ou égal à 200 sur 100 épisodes consécutifs, avec une démonstration claire de stabilité (Intervalle de Confiance à 95% et analyse de l'écart-type). 
Ce projet met également en lumière notre démarche scientifique pour optimiser les hyperparamètres d'un modèle de Reinforcement Learning.

## 2. Fonctionnement du dqn

Le Deep Q-Network (DQN) est un algorithme qui apprend par essais et erreurs. Dans un environnement simple, une IA peut apprendre en mémorisant un tableau de règles par cœur ("si je suis dans cette position exacte, je fais cette action"). Mais dans LunarLander, les combinaisons de vitesse, d'angle et de position sont infinies : c'est impossible à mémoriser. Le DQN résout ce problème en utilisant un réseau de neurones. Ce réseau agit comme un évaluateur : il lit les capteurs du vaisseau en temps réel et donne une note à chacune des 4 actions possibles (moteur gauche, droit, principal, ou rien). Le vaisseau choisit alors simplement l'action avec la meilleure note.
Nous avons choisi cet algorithme précisément pour sa capacité à "généraliser". Grâce à son réseau de neurones, même si le vaisseau se retrouve dans une situation avec un angle de chute qu'il n'a jamais vu pendant son entraînement, il comprendra la physique globale du problème et saura en déduire l'action logique pour se redresser et atterrir.


## 3. L'amelioration de notre model.

### Phase 1: Les Fondations (MLP)
Durant cette phase nous avons simplement passé du temps à nous familiarisé avec l'envvironnement et le fonctionnement du lunar lander v3.
On a implémenter le reseau de neurones (MLP) avec une architecture de base avec des couches denses et une activation ReLU pour traiter les 8 variables d'état de l'environnement.
On a mis en place l'epsilon-Greedy avec un taux de décroissance (notre epsilon decay) pour transitionner progressivement vers l'exploitation des connaissances de notre model.
Et finalement on a mis en place le replayer buffer pour donner une memoire d'experience à notre model afin de briser la corrélation temporelle des séquences d'apprentissage, en utilisant des batchs de taille 64.

### Phase 2: Stabilisation de l'Apprentissage (Q-Target)
On a mis en place le target Network pour marquer la séparation entre le réseau principal et le réseau cible pour éviter l'effet de cible mouvante lors des mises à jour du model. On a commencé avec une mise à jour soft afin de lissé considérablement la courbe d'apprentissage de notre model avec un tau de (0.001).

### Phase 3 : L'Évolution Mathématique (Double DQN)
Afin de tester d'améliorer notre model nous avons aussi essayer d'utiliser le DoubleDqn ça consiste à utiliser notre Target Network comme un deuxieme Qnetwork afin que les 2 s'entrainent mutuellement en s'entraidant, Le réseau local sélectionne la meilleure action, tandis que le réseau cible évalue sa valeur.
Nous avons experimenté cette methode afin de voir si les resultats de cette derniere était plus intéressant.
Bien que la théorie promette une meilleure stabilité, nos recherches d'hyperparamètres sur cette architecture précise ont montré une convergence plus chaotique, nous poussant à privilégier le simple DQN.
Suite à ça nous avons donc décidé de continuer à nous intéresser plus au dqn pour les etapes à suivre tout en continuant à tester le double dqn.

### Phase 4 : Recherche Scientifique et Ablation (Grid Search)
Afin de justifier techniquement nos choix architecturaux, une recherche systématique
d'hyperparamètres (Grid Search) a été exécutée. Cette phase a permis d'identifier et de corriger
l'overfitting. Tout en cherchant les meilleurs parametres possible pour notre model.

Hyperparamètre     |    Valeurs testées (Soft Update)              |     Valeurs testées (Hard Update)
-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------

Episodes           |  "400, 500, 600, 800, 1000, 1200, 1500"       |  "400, 500, 600, 800, 1000, 1200, 1500"
-------------------------------------------------------------------------------------------------------------
Learning Rate      |        "1e-3, 7.5e-4, 5e-4"                   |                "1e-3, 5e-4"
-------------------------------------------------------------------------------------------------------------
Batch Size         |              "64, 128"                        |                 "64, 128"
-------------------------------------------------------------------------------------------------------------
Hidden Size        |           "64, 128, 256"                      |                "64, 128"
-------------------------------------------------------------------------------------------------------------
Buffer Capacity    |   "10 000, 30 000, 50 000, 100 000"           |        "10 000, 30 000, 50 000"
-------------------------------------------------------------------------------------------------------------
Epsilon Decay      |             "0.99, 0.985"                     |              "0.995, 0.99"
-------------------------------------------------------------------------------------------------------------
Tau (τ)            |          "1e-3, 3e-3, 5e-3"                   |              Non applicable
-------------------------------------------------------------------------------------------------------------
Target Update Frequency      |     Non applicable                  |            "100, 200, 500, 1000"
-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------

* vous pouvez retrouvez le résultat de tputes ces experimentations dans le dossier "documentation\dqn\grid_search\" dans les fichiers leaderboard_final.json (premier gridsearch) leaderboard_analytics.csv (second gridsearch).

Au final nous avons décidé de garder les hyperparametres suivant car c'est ceux qui nous permettaient de garder la meilleure combinaison entre performance et stabilité tout en s'entrainant rapidement tout.
Nous avons choisi d'entrainer sur 400 episodes car ce nombre permet d'offrir le meilleur compromis entre temps de calcul et performance de convergence.

hyperparameters:
  learning_rate: 0.00075
  batch_size: 64
  hidden_size: 64
  epsilon_decay: 0.99
  update_type: "soft"
  tau: 0.001 
  gamma: 0.99
  buffer_capacity: 30000
  epsilon_start: 1.0
  epsilon_min: 0.01
  episodes: 400
