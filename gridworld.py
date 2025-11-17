import numpy as np
import random

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (9, 9)

        # Obstáculos fijos (simulan estantes)
        self.obstacles = {
            (2, 2), (2, 3), (2, 4),
            (5, 5), (6, 5), (7, 5),
            (3, 7), (4, 7)
        }

        # Zonas congestionadas
        self.congestion = {
            (1, 4), (1, 5), (1, 6),
            (6, 2), (7, 2)
        }

        self.reset()

    # -------------------------------------------------------------------

    def reset(self):
        """Reinicia el entorno al estado inicial."""
        self.agent_pos = self.start
        return self.agent_pos

    # -------------------------------------------------------------------

    def step(self, action):
        """
        Acciones:
        0 = arriba
        1 = abajo
        2 = izquierda
        3 = derecha
        """

        r, c = self.agent_pos
        nr, nc = r, c

        # Movimiento según acción
        if action == 0 and r > 0:
            nr -= 1
        elif action == 1 and r < self.size - 1:
            nr += 1
        elif action == 2 and c > 0:
            nc -= 1
        elif action == 3 and c < self.size - 1:
            nc += 1

        next_pos = (nr, nc)

        # Si intenta entrar a un obstáculo → se queda en el mismo sitio
        if next_pos in self.obstacles:
            next_pos = self.agent_pos

        # Recompensa base
        reward = -1  # penaliza tiempo

        # Penalización adicional por congestión
        if next_pos in self.congestion:
            reward -= 4

        # Recompensa final si llega a la meta
        if next_pos == self.goal:
            reward = 100

        # Actualizar estado del agente
        self.agent_pos = next_pos

        done = (next_pos == self.goal)

        return next_pos, reward, done
