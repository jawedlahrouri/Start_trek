# Documentation Train & Agents : Projet Start Trek

## 1. Le rôle du script
Le script `train.py` est le chef d'orchestre de notre projet. C'est lui qui initialise l'environnement LunarLander, lance les entraînements de nos modèles, et s'occupe de récupérer les statistiques pour nos analyses. Au début, on avait un script très basique, mais on a dû l'améliorer au fur et à mesure pour respecter les critères du sujet, comme la reproductibilité et l'enregistrement de métriques précises et surtout pour faciliter nos experimentations grâce aux fichiers de configs.

## 2. L'évolution de notre code

### Phase 1 : Le code en dur
Au tout début de nos expérimentations, on avait mis tous nos hyperparamètres directement dans le fichier `train.py` (comme le learning rate, les épisodes, etc.). C'était suffisant pour faire décoller le vaisseau la première fois, mais c'était très compliqué pour faire des tests (notre Grid Search) car on devait modifier le code source à chaque nouvelle idée.

### Phase 2 : Le passage au YAML
Pour faciliter nos tests, on a décidé de sortir toutes nos variables dans des fichiers `.yaml` comme le demandait le sujet. On a créé un fichier `config_loader.py` qui lit le fichier de configuration et envoie les informations au `train.py`. Grâce à ça, on a pu automatiser nos recherches et lancer plein d'entraînements avec des paramètres différents sans jamais toucher au code principal.

### Phase 3 : L'architecture Orientée Objet (Les Agents)
Quand on a voulu tester d'autres algorithmes comme le DoubleDQN et le PPO, notre `train.py` devenait beaucoup trop lourd. On a donc décidé d'utiliser la programmation orientée objet. 
On s'est donc inspiré de nos autres projet epitech pour créé une classe `BaseAgent` qui sert d'interface de base, et on a séparé chaque algorithme dans son propre fichier (`dqn_agent.py`, `doubledqn_agent.py`, `ppo_agent.py`). Maintenant, le `train.py` lit simplement le nom de l'algo dans le YAML et charge l'agent correspondant. 

## 3. L'architecture finale du train.py

Aujourd'hui, notre script d'entraînement est robuste et intègre tout ce qui est demandé par le barème :
- **La reproductibilité :** On a ajouté une fonction `set_global_seed(seed)` qui bloque complètement l'aléatoire de python et de pytorch. Ça nous garantit de pouvoir relancer un entraînement et d'avoir exactement les mêmes résultats (One-click repro).
- **Les vidéos et l'instrumentation :** On a intégré TensorBoard avec `SummaryWriter` pour tracer nos courbes, et `RecordVideo` pour sauvegarder une vidéo MP4 du vaisseau tous les 100 épisodes.
- **Les logs de fin :** Le script analyse chaque fin de partie pour savoir exactement pourquoi elle s'est arrêtée (crash, out-of-view, sleep ou safe-landing) et l'envoie dans nos statistiques.

## 4. Nos différents agents

Pour que `train.py` puisse les utiliser facilement, nos agents partagent tous les mêmes méthodes de base comme `select_action`, `step`, `update` et `get_logs`.

- **DQNAgent :** C'est notre modèle principal. Il apprend en temps réel à chaque étape de l'environnement (Off-Policy).
- **DoubleDQNAgent :** Une variante qu'on a implémentée pour essayer d'améliorer la stabilité de l'apprentissage.
- **PPOAgent :** Contrairement au DQN, le PPO fonctionne différemment (On-Policy). Il accumule les données pendant toute la partie et fait sa mise à jour (update) d'un seul coup à la fin de l'épisode. On a donc ajouté une condition dans `train.py` pour s'adapter à cette logique spécifique sans casser le reste.
