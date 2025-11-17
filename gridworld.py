import numpy as np
import random

class GridWorld:
    def __init__(self, filas=6, columnas=6):
        self.filas = filas
        self.columnas = columnas

        # 0 = libre
        # -1 = obstáculo
        # -2 = penalización por congestión
        # 1 = meta
        self.mapa = np.array([
            [0, 0, 0, 0, -2, 1],
            [0, -1, -1, 0, -2, 0],
            [0, 0, 0, 0, 0, 0],
            [-2, -2, 0, -1, 0, 0],
            [0, 0, 0, 0, -2, 0],
            [0, 0, -1, 0, 0, 0]
        ])

        self.estado_inicial = (0, 0)
        self.meta = (0, 5)

        self.reset()

    def reset(self):
        self.agente = self.estado_inicial
        return self.agente

    def get_actions(self):
        return ["arriba", "abajo", "izquierda", "derecha"]

    def step(self, accion):
        fila, col = self.agente

        if accion == "arriba": fila -= 1
        elif accion == "abajo": fila += 1
        elif accion == "izquierda": col -= 1
        elif accion == "derecha": col += 1

        # Validar límites
        if fila < 0 or fila >= self.filas or col < 0 or col >= self.columnas:
            return self.agente, -5, False   # Penalización por golpear pared

        # Validar obstáculo
        if self.mapa[fila, col] == -1:
            return self.agente, -10, False

        self.agente = (fila, col)

        # Recompensas
        if self.agente == self.meta:
            return self.agente, 20, True

        if self.mapa[fila, col] == -2:
            return self.agente, -3, False

        return self.agente, -1, False