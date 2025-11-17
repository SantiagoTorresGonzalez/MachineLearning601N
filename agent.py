import numpy as np

class QAgent:
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((state_dim, state_dim, action_dim))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])

        self.q_table[state[0], state[1], action] += self.lr * (
            reward + self.gamma * max_next_q - current_q
        )
