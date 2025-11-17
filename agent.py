import numpy as np
import random

class QAgent:
    def __init__(self, grid_size=10, action_dim=4, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        """
        Agente Q-Learning para un GridWorld de tamaño grid_size x grid_size.

        - q_table: matriz [fila][columna][acción]
        - epsilon: exploración inicial
        - epsilon_decay: reduce la exploración cada episodio
        """

        self.grid_size = grid_size
        self.action_dim = action_dim

        # Q-table inicializada en ceros
        self.q_table = np.zeros((grid_size, grid_size, action_dim))

        # Hiperparámetros
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # --------------------------------------------------
    # ELECCIÓN DE ACCIÓN (explorar/explotar)
    # --------------------------------------------------
    def choose_action(self, state):
        """
        Selección de acción usando ε-greedy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)   # acción aleatoria

        row, col = state
        return np.argmax(self.q_table[row, col])            # mejor acción

    # --------------------------------------------------
    # ACTUALIZACIÓN DE LA Q-TABLE
    # --------------------------------------------------
    def update(self, state, action, reward, next_state):
        """
        Actualización Q-Learning:
        Q(s,a) ← Q(s,a) + lr [ r + γ max(Q(s',a)) − Q(s,a) ]
        """

        r, c = state
        nr, nc = next_state

        current_q = self.q_table[r, c, action]
        max_next_q = np.max(self.q_table[nr, nc])

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[r, c, action] = new_q

    # --------------------------------------------------
    # REDUCCIÓN DE EXPLORACIÓN POR EPISODIO
    # --------------------------------------------------
    def reduce_epsilon(self):
        """
        Reduce la exploración progresivamente para mejorar estabilidad.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay