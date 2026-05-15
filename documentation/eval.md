# Documentation : Comment fonctionne `eval.py`

## Pourquoi `eval.py` ?

Le script `eval.py` est conçu pour évaluer les performances d'un agent entraîné sur l'environnement LunarLander (ou tout autre environnement compatible avec Gymnasium). Ses principaux objectifs sont :

- **Charger un modèle sauvegardé** : Il peut charger les poids de divers agents comme **DQN**, **DoubleDQN**, **PPO**.
- **Évaluation robuste** : Il exécute des épisodes d'évaluation sur plusieurs graines pour garantir que les résultats sont statistiquement significatifs et non dus au hasard.
- **Statistiques détaillées** : Il génère des métriques de performance clés, notamment le score moyen, l'écart type et l'intervalle de confiance à 95 %.
- **Enregistrement des données** : Il sauvegarde les résultats détaillés, épisode par épisode, dans un fichier CSV pour une analyse plus approfondie.
- **Analyse visuelle** : Il enregistre des vidéos des épisodes d'évaluation, permettant une inspection visuelle du comportement de l'agent.

## Comment fonctionne `eval.py` ?

Le script suit un processus clair, étape par étape :

1.  **Chargement de la configuration** :
    -   Le script nécessite un fichier de configuration YAML comme argument de ligne de commande. Ce fichier spécifie l'algorithme, les paramètres de l'environnement, les hyperparamètres et les graines à utiliser pour l'évaluation.
2.  **Chargement du modèle** :
    -   Le modèle entraîné est chargé à partir du chemin spécifié dans le fichier de configuration. Un mécanisme de secours est inclus pour rechercher le modèle dans un répertoire par défaut `assets/agents_save/` s'il n'est pas trouvé au chemin principal.
3.  **Configuration de l'environnement** :
    -   L'environnement Gymnasium est créé en utilisant les paramètres du fichier de configuration (par exemple, la gravité, le vent).
    -   Si l'enregistrement vidéo est activé, l'environnement est encapsulé avec `RecordVideo` pour sauvegarder chaque épisode.
4.  **Boucle d'évaluation** :
    -   Le script itère sur chaque graine spécifiée. Par défaut, il exécute 100 épisodes par graine.
    -   Dans chaque épisode :
        -   L'agent agit de manière gloutonne, choisissant toujours l'action avec la plus grande valeur Q du modèle chargé.
        -   Les scores, les causes de fin d'épisode (par exemple, `crash`, `atterrissage-réussi`, `hors-champ`) et le nombre de pas sont enregistrés.
        -   Les résultats de l'épisode sont ajoutés au fichier CSV.
5.  **Statistiques et résumé** :
    -   Une fois tous les épisodes terminés, le script calcule et affiche un résumé final.
    -   Cela inclut des statistiques globales et par graine (moyenne, écart type, IC).
    -   Un histogramme en mode texte montre la distribution des scores.
    -   Une répartition des causes de fin et le taux de réussite global sont imprimés.
    -   Enfin, il rappelle à l'utilisateur où les résultats CSV et les fichiers vidéo ont été sauvegardés.

## Points clés à retenir

-   `eval.py` est l'outil central pour mesurer objectivement les performances de vos agents RL.
-   Il assure la **reproductibilité** grâce à l'utilisation de plusieurs graines et à l'enregistrement systématique des données.
-   Il fournit un **rapport complet** (CSV, vidéos, statistiques de la console) pour faciliter l'analyse et la comparaison de différents modèles.
-   Utilisation : `python eval.py <chemin_vers_config.yaml>`.
-   Le script est facilement extensible pour prendre en charge d'autres algorithmes ou environnements.
