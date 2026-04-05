import numpy as np
from collections import defaultdict


# ε-greedy action selection
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    return np.argmax(Q[state])


# ---------------------------
# TRAIN (MC CONTROL)
# ---------------------------
def train_mc(env, num_episodes=5000, gamma=1.0, epsilon=0.1,
             track_training=True, num_checkpoints=100):

    Q = defaultdict(lambda: np.zeros(2))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    training_rewards = []
    training_episodes = []

    checkpoint_interval = num_episodes // num_checkpoints
    reward_buffer = []

    for episode_num in range(1, num_episodes + 1):
        episode = []
        state, _ = env.reset()

        done = False
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode.append((state, action, reward))
            state = next_state

        # store final reward
        reward_buffer.append(episode[-1][2])

        # compute returns
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

        # tracking for plots
        if track_training and episode_num % checkpoint_interval == 0:
            training_rewards.append(np.mean(reward_buffer))
            training_episodes.append(episode_num)
            reward_buffer = []

    # greedy policy
    policy = {s: np.argmax(a) for s, a in Q.items()}

    return policy, training_rewards, training_episodes


# ---------------------------
# TEST (GREEDY POLICY)
# ---------------------------
def test_mc(env, policy, num_episodes=1000):
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy.get(state, 0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        rewards.append(reward)

    avg_reward = np.mean(rewards)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards)

    return avg_reward, win_rate, rewards