from gridworld import GridWorld
from agent import QAgent

def run_training(episodes=50):
    env = GridWorld(size=4)
    agent = QAgent(state_dim=4, action_dim=4)

    history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            state = next_state
            steps += 1
            total_reward += reward

        history.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": round(total_reward, 3)
        })

    return history
