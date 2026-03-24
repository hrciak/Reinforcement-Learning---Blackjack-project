'''
This will be for Temporal difference algorithms such as Sarsa and Q-Learning
'''
# 1. SARSA (On Policy)


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 2.Q-learning TDalg (Off Policy)

import numpy as np

def encode_state(state):
    """convert (player_sum, dealer_card, usable_ace) tuple to single integer index (0-703)"""
    player_sum, dealer_card, usable_ace = state
    return player_sum * 22 + dealer_card * 2 + usable_ace

def epsilon_greedy(Q, state, epsilon, env): #Q-table, what state are we in, what proportion, environment
    """select action using epsilon-greedy strategy"""
    state_idx = encode_state(state)  # Convert tuple to integer index
    if np.random.rand() < epsilon: #10% of the time chooses a random action, 90% following the policy (off policy based algo)
        return env.action_space.sample()
    else:
        return np.argmax(Q[state_idx]) #What is teh best action I can take based on our Q table
    
def q_learning(env, num_episodes, alpha = 0.1, gamma = 0.99, epsilon = 0.1, track_training=False): #environment, number of repetitions of the game, learning rate, decay (importance of current actions vs future actions), proportion of the time choosing a random action  
    """q-learning algorithm for blackjack env
    
    if track_training=True, returns (Q, training_metrics)
    else returns just Q
    """
    Q = np.zeros((32 * 11 * 2, env.action_space.n))  #704 states, 2 actions => 32 max sum of players hand, 11 max of dealers shown card, 0 or 1 (bool state) for ace on players hand

    training_rewards = []
    training_episodes_list = []
    checkpoint_interval = max(1, num_episodes // 50)  # Create ~50 checkpoints
    episode_reward_buffer = []

    for episode in range(num_episodes):
        state, _ = env.reset() #starting the game
        done = False
        episode_reward = 0
        
        while not done:
            state_idx = encode_state(state)  # Convert tuple to integer
            action = epsilon_greedy(Q, state, epsilon, env) #action to take at the state (random or from Q-table)
            next_state, reward, terminated, truncated, _ = env.step(action) #Taking a step inside this env with this action
            done = terminated or truncated
            episode_reward += reward
    
            #only use bootstrap (max of next Q-values) if episode didn't end
            if done:
                Q[state_idx, action] = Q[state_idx, action] + alpha * (reward - Q[state_idx, action])
            else:
                next_state_idx = encode_state(next_state)  # Convert tuple to integer
                Q[state_idx, action] = Q[state_idx, action] + alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action])
            
            state = next_state
        
        # Track training progress at intervals
        if track_training:
            episode_reward_buffer.append(episode_reward)
            if (episode + 1) % checkpoint_interval == 0:
                avg_reward = np.mean(episode_reward_buffer)
                training_rewards.append(avg_reward)
                training_episodes_list.append(episode + 1)
                episode_reward_buffer = []
    
    if track_training:
        return Q, training_rewards, training_episodes_list
    else:
        return Q



#Test Q-LEarning implementation (for report + visuals)
def test_q_learning(env, Q, num_episodes=1000):
    """evaluate a trained Q-learning policy using greedy action selection.
    
    returns three variables:
        1. avg_reward: average reward over episodes
        2. success_rate: proportion of episodes with positive reward (win rate)
        3. rewards: list of episode rewards
    """
    
    rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_idx = encode_state(state)
            
            # Greedy action (no exploration)
            action = np.argmax(Q[state_idx])
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        # In Blackjack: reward > 0 means win
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
    
    avg_reward = np.mean(rewards)
    win_rate = wins / num_episodes

    if Q.shape != (32 * 11 * 2, env.action_space.n):
        raise ValueError("Q-table shape doesn't match environment")
    
    return avg_reward, win_rate, rewards


 