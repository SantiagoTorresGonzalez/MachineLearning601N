import matplotlib.pyplot as plt

def entrenar(env, agent, episodios=500):
    historial = []

    for ep in range(episodios):
        estado = env.reset()
        total_reward = 0

        done = False
        while not done:
            accion = agent.choose_action(estado)
            nuevo_estado, recompensa, done = env.step(accion)
            agent.update(estado, accion, recompensa, nuevo_estado)

            estado = nuevo_estado
            total_reward += recompensa

        agent.decay()
        historial.append(total_reward)

    return historial