'''
Plots for convergence - while training & test evaluations

PREREQUISITES FOR ALGORITHM IMPLEMENTATIONS TO USE THESE PLOTS:
================================================================

Any RL algorithm (Q-Learning, SARSA, Monte Carlo, etc.) must satisfy these requirements
for the plotting functions to work correctly and produce meaningful comparisons.

1. TRAINING FUNCTION REQUIREMENTS:
   - Must track training metrics if optional tracking is enabled
   - Must return `training_rewards`: list of AVERAGE EPISODE REWARDS at regular intervals
   - Must return `training_episodes`: list of EPISODE NUMBERS corresponding to each checkpoint
   - Both lists must be same length
   - Rewards at each checkpoint = mean(rewards over that interval)
   - Intervals should be uniform (e.g., every 50 episodes for 5000 total episodes)

2. TEST FUNCTION REQUIREMENTS:
   - Must return `rewards`: list of individual episode rewards from test phase
   - Each element = scalar [-1, 0, or 1] representing single episode outcome
   - List length = number of test episodes (typically 1000)
   - Test evaluation uses GREEDY policy (NO exploration/randomness)
   - Test must use fresh environment episodes (not contaminated by training)

3. REWARD SCALE COMPATIBILITY:
   - Training rewards must be on same scale as test rewards
   - Both must be directly comparable across algorithms
   - For Blackjack: rewards are -1 (loss), 0 (draw), +1 (win)
   - NO reward normalization or scaling - keep raw values

4. EPISODIC STRUCTURE:
   - Algorithm must be episodic (episodes have clear termination)
   - Each training episode produces one reward signal
   - Training checkpoints aggregate multiple episodes into average

5. FUNCTION SIGNATURES:
   - Training function: train_algo(env, num_episodes, track_training=True, num_checkpoints=100)
   - Test function: test_algo(env, policy/Q_table) -> (avg_reward, success_rate, rewards)
   - Both functions must be independent (test doesn't affect training state)

CRITICAL: Deviations from these prerequisites will cause:
- Incompatible plot scales between algorithms
- Meaningless convergence comparisons
- Incorrect evaluation metrics
'''

import numpy as np
import matplotlib.pyplot as plt


#Training Efficiency Plots (measure convergence during training)
# PREREQUISITE: train_algo() must return (policy, training_rewards, training_episodes)
# WHERE: training_rewards = list of avg rewards per checkpoint
#        training_episodes = list of episode numbers at each checkpoint

def plot_training_curve(episodes, rewards, algorithm_name="Q-Learning"):
    """plot average reward per checkpoint during training to show convergence
    
    episodes: list of episode numbers at checkpoints
    rewards: list of average rewards at checkpoints
    algorithm_name: name of algorithm for title (default: "Q-Learning")
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, marker='o', linewidth=2, markersize=4, color='blue')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.fill_between(episodes, rewards, alpha=0.2)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"{algorithm_name} Training Curve (Convergence)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_training_efficiency_summary(episodes, rewards, algorithm_name="Q-Learning"):
    """plot simpler training efficiency metric showing learning progress
    
    episodes: list of episode numbers at checkpoints
    rewards: list of average rewards at checkpoints
    algorithm_name: name of algorithm for title (default: "Q-Learning")
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Show improvement from first to last checkpoint
    initial_reward = rewards[0]
    final_reward = rewards[-1]
    improvement = final_reward - initial_reward
    
    ax.plot(episodes, rewards, marker='s', linewidth=2.5, markersize=5, color='darkblue', label='Avg Reward per Interval')
    ax.axhline(y=initial_reward, color='orange', linestyle=':', alpha=0.7, label=f'Initial: {initial_reward:.4f}')
    ax.axhline(y=final_reward, color='green', linestyle=':', alpha=0.7, label=f'Final: {final_reward:.4f}')
    ax.fill_between(episodes, rewards, alpha=0.15, color='blue')
    
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Reward (Checkpoint Interval)")
    ax.set_title(f"{algorithm_name} Training Efficiency (Improvement: {improvement:+.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    
    print(f"{algorithm_name} Training Efficiency Summary:")
    print(f"  Initial reward (checkpoint 1): {initial_reward:.4f}")
    print(f"  Final reward (checkpoint {len(episodes)}): {final_reward:.4f}")
    print(f"  Total improvement: {improvement:+.4f}")
    print(f"  Improvement rate: {improvement/len(episodes)*100:.2f}% per checkpoint")


#Test Performance Plots (evaluate policy after training)
# PREREQUISITE: test_algo() must return (avg_reward, win_rate, rewards)
# WHERE: rewards = list of individual episode rewards [-1, 0, or 1]
#        Length = number of test episodes (typically 1000)
def plot_cumulative_rewards(rewards, algorithm_name="Q-Learning"):
    """plot cumulative rewards over episodes
    
    rewards: list of episode rewards
    algorithm_name: name of algorithm for title (default: "Q-Learning")
    """
    cumulative = np.cumsum(rewards)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative, linewidth=2, color='blue')
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algorithm_name} Cumulative Performance")
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_outcome_distribution(rewards, algorithm_name="Q-Learning"):
    """plot histogram of outcomes (bar chart)
    
    rewards: list of episode rewards
    algorithm_name: name of algorithm for title (default: "Q-Learning")
    """
    wins = sum(1 for r in rewards if r > 0)
    losses = sum(1 for r in rewards if r < 0)
    draws = sum(1 for r in rewards if r == 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = ['Win', 'Draw', 'Loss']
    counts = [wins, draws, losses]
    colors = ['green', 'gray', 'red']
    ax.bar(outcomes, counts, color=colors, alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title(f"{algorithm_name} Policy Outcomes")
    for i, v in enumerate(counts):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
    plt.show()

def plot_test_results(rewards, algorithm_name="Q-Learning"):
    """plot rewards with moving average to show policy performance trend
    
    rewards: list of episode rewards
    algorithm_name: name of algorithm for title (default: "Q-Learning")
    """
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, label='Episode Reward')
    ax.plot(range(len(moving_avg)), moving_avg, label=f'Moving Average (window={window})', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{algorithm_name} Policy Performance (Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


