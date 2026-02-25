# 🃏 Reinforcement Learning — Blackjack

> **SOW-BKI258 Reinforcement Learning** · Radboud University · 2025–2026

A comparative study of tabular Reinforcement Learning algorithms applied to the **Blackjack-v1** environment from [Gymnasium](https://gymnasium.farama.org/). We implement and evaluate Dynamic Programming, Monte Carlo, and Temporal Difference methods, and analyse their performance in a written report inside the Jupyter Notebook.




## 📁 Project Structure

```
.
├── notebook.ipynb       # Main Jupyter Notebook (environment + report)
├── dp.py                # Dynamic Programming algorithms
├── mc.py                # Monte Carlo algorithms
├── td.py                # Temporal Difference algorithms
└── requirements.txt     # Project dependencies
```



## 🎰 Environment

We use the **Blackjack-v1** environment from Gymnasium's *Toy Text* collection. It is an episodic card game with a discrete state and action space, well-suited for tabular RL methods.

```python
import gymnasium as gym
env = gym.make("Blackjack-v1")
obs, info = env.reset()
```

- **State space:** (player sum, dealer showing card, usable ace)
- **Action space:** `0` = stick, `1` = hit
- **Goal:** Reach a hand value closer *(or equal to)* to 21 than the dealer without going bust



## 🤖 Implemented Algorithms

### Dynamic Programming (`dp.py`)
Requires full knowledge of the environment's transition dynamics.

| Algorithm | Method |
|---|---|
| Policy Evaluation | Iterative, computes state values V(s) |
| Policy Improvement | Greedy policy update |
| Value Iteration | Combined evaluation + improvement |

### Monte Carlo (`mc.py`)
Learns from complete episodes without a model of the environment.

| Algorithm | Method |
|---|---|
| MC Prediction | First-visit, evaluates action values Q(s,a) |
| MC Control | Exploring Starts *or* ε-Greedy strategy |

### Temporal Difference (`td.py`)
Bootstraps from incomplete episodes, learning at every step.

| Algorithm | Method |
|---|---|
| TD(0) | One-step value prediction |
| SARSA | On-policy ε-Greedy control |
| Q-Learning | Off-policy ε-Greedy control |

> A **random baseline agent** is also included for benchmarking.



## 📊 Report

The report is written inside `notebook.ipynb` and covers:

- **Introduction** — environment description, agent objective, research question
- **Dynamic Programming** — algorithm descriptions, parameter sensitivity (γ, θ), policy plots
- **Monte Carlo** — prediction and control results, hyperparameter tuning
- **Temporal Difference** — SARSA vs Q-Learning comparison, ε decay strategy
- **Comparison & Discussion** — MC vs TD metrics (cumulative reward, RMSE, sample efficiency)
- **Conclusion** — key findings and algorithm trade-offs




## ⚙️ Setup

```bash
# Clone the repository
git clone "https://github.com/hrciak/Reinforcement-Learning---Blackjack-project"
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** includes at minimum:
```
gymnasium
numpy
matplotlib
jupyter
```


## 👥 Workgroup

Made by Workgroup **40**: Joudi Jomah, Razan Mushmush, Paraskevi Paliou, Jozef Hrcka 
[Radboud University, Period 3, 2025–2026]