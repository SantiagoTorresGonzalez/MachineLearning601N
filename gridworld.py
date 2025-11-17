import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        return self.agent_pos

    def step(self, action):
        # 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        done = self.agent_pos == [self.size - 1, self.size - 1]
        reward = 1 if done else -0.01

        return self.agent_pos, reward, done
