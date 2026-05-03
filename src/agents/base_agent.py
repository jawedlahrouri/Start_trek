from abc import ABC, abstractmethod

class BaseAgent(ABC):
    #c'est la classe de base de tous les agents (c une interface en cpp stv)
    def __init__(self, state_size, action_size, hypers):
        self.state_size = state_size
        self.action_size = action_size
        self.hypers = hypers

    @abstractmethod
    def select_action(self, state, current_ep):
        #la methdode qui choisit l'action
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        #la méthode qui mémorise la transition dans la mémoire
        pass

    @abstractmethod
    def update(self):
        #la méthode qui effectue la maj du modele
        pass

    @abstractmethod
    def get_logs(self):
        #Retourne les infos pour Tensorboard
        pass

    @abstractmethod
    def save(self, path):
        #Sauvegarde le modèle
        pass
