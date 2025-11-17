from gridworld import GridWorld
from agent import QAgent
import numpy as np

def run_training(episodes=200):
    env = GridWorld(size=10)
    agent = QAgent(grid_size=10)

    rewards_history = []
    eps_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # Reducir la exploración al final de cada episodio
        agent.reduce_epsilon()

        # Guardar estadísticas
        rewards_history.append(total_reward)
        eps_history.append(agent.epsilon)

    # Devolver datos para Plotly
    return {
        "rewards": rewards_history,
        "epsilon": eps_history
    }