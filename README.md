Project Overview

This project implements and compares multiple tabular Reinforcement Learning (RL) algorithms using the Blackjack environment from Gymnasium (Toy Text domain).

The objective is to analyze how different RL methods—Dynamic Programming (DP), Monte Carlo (MC), and Temporal Difference (TD)—learn optimal policies in a stochastic, episodic card game setting.

Blackjack is a finite Markov Decision Process (MDP) with:

Discrete state space

Discrete action space

Episodic structure

Stochastic transitions

The agent’s goal is to maximize expected cumulative reward by learning when to hit or stick based on the player’s sum, dealer’s visible card, and usable ace indicator.

Environment: Blackjack

The environment is instantiated via:

env = gym.make("Blackjack-v1")
State Representation

Each state is represented as a tuple:

Player’s current sum (12–21)

Dealer’s visible card (1–10)

Usable ace (True/False)

Action Space

0: Stick

1: Hit

Reward Structure

+1 → Player wins

0 → Draw

−1 → Player loses

Key Characteristics

Episodic environment (terminal when player sticks or busts)

Stochastic transitions (card draws)

Suitable for tabular methods (manageable state space)

Code Structure

The implementation is modular and consists of:

dp.py → Dynamic Programming algorithms

mc.py → Monte Carlo algorithms

td.py → Temporal Difference algorithms

notebook.ipynb → Environment setup, experiments, plots, and report

The notebook calls functions from the algorithm files to run experiments and visualize results.

Core Environment Functions

The Blackjack environment follows the standard Gymnasium API:

1. reset()

Initializes a new episode.

state, info = env.reset()
2. step(action)

Executes an action and returns:

next_state, reward, terminated, truncated, info = env.step(action)
3. render() (optional)

Displays current game state.

Additional Utility Functions (Custom)

To support RL algorithms, the following utility functions are integrated:

Episode Handling

generate_episode(policy)
Generates a full episode: (state, action, reward) sequence.

is_terminal(state)
Checks if the episode has ended.

Policy Utilities

epsilon_greedy_policy(Q, state, epsilon)
Samples action under ε-greedy exploration.

greedy_policy(Q, state)
Returns action with highest Q-value.

Value Conversions

q_to_v(Q)
Converts action-value function to state-value function.

extract_policy(Q)
Extracts deterministic policy from Q-values.

Evaluation & Metrics

compute_cumulative_reward(policy, n_episodes)

compute_rmse(Q, Q_optimal)

plot_learning_curve(rewards)

plot_policy(policy)

plot_value_function(V)

Implemented Reinforcement Learning Algorithms
1. Dynamic Programming (dp.py)

policy_evaluation(policy, env, gamma, theta)

policy_improvement(V, env, gamma)

policy_iteration(env, gamma, theta)

value_iteration(env, gamma, theta)

Purpose:
Provides ground truth optimal values and policies for comparison.

2. Monte Carlo Methods (mc.py)

mc_prediction(env, policy, gamma, n_episodes)

mc_control_es(env, gamma, n_episodes)

mc_control_epsilon_greedy(env, gamma, epsilon, n_episodes)

Characteristics:

Model-free

Episode-based updates

High variance, unbiased estimates

3. Temporal Difference Learning (td.py)

td_zero(env, policy, alpha, gamma, n_episodes)

sarsa(env, alpha, gamma, epsilon, n_episodes)

q_learning(env, alpha, gamma, epsilon, n_episodes)

Characteristics:

Bootstrapping updates

Online learning

Lower variance than MC

4. Baseline Agent

random_policy()
Used as performance benchmark.

Evaluation Metrics

Algorithms are compared using:

Cumulative reward per episode

Learning curves

RMSE vs DP optimal values

Sample efficiency

Final policy comparison

Project Objective

The primary objective is to:

Implement tabular RL algorithms.

Analyze their convergence behavior in Blackjack.

Compare learning efficiency and policy quality.

Evaluate strengths and limitations of DP, MC, and TD methods.

Summary

This project investigates how different reinforcement learning paradigms perform in the Blackjack environment from Gymnasium.

Dynamic Programming provides optimal reference values, while Monte Carlo and Temporal Difference methods learn through interaction. Their convergence speed, stability, and final policy performance are systematically compared using quantitative metrics and visualizations.