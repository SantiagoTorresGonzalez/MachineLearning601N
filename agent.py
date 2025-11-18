import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, alfa=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.actions = actions
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = {}  # Diccionario {(estado): {accion: valor}}

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
        return self.Q[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        Q_state = self.get_Q(state)
        return max(Q_state, key=Q_state.get)

    def update(self, state, action, reward, next_state):
        Q_state = self.get_Q(state)
        Q_next = self.get_Q(next_state)

        Q_state[action] += self.alfa * (reward + self.gamma * max(Q_next.values()) - Q_state[action])

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay