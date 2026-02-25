# 🃏 Reinforcement Learning — Blackjack

> **SOW-BKI258 Reinforcement Learning** · Radboud University · 2025–2026

A comparative study of tabular Reinforcement Learning algorithms applied to the **Blackjack** environment from [Gymnasium](https://gymnasium.farama.org/). We implement and evaluate Dynamic Programming, Monte Carlo, and Temporal Difference methods, and analyse their performance in a written report inside the Jupyter Notebook.

---

## 📁 Project Structure

```
.
├── notebook.ipynb       # Main Jupyter Notebook (environment + report)
├── dp.py                # Dynamic Programming algorithms
├── mc.py                # Monte Carlo algorithms
├── td.py                # Temporal Difference algorithms
└── requirements.txt     # Project dependencies
```

---

## 🎰 Environment

We use the **Blackjack-v1** environment from Gymnasium's *Toy Text* collection — a classic episodic card game with a discrete state and action space, well-suited for tabular RL methods.

```python
import gymnasium as gym
env = gym.make("Blackjack-v1")
obs, info = env.reset()
```

- **State space:** (player sum, dealer showing card, usable ace)
- **Action space:** `0` = stick, `1` = hit
- **Goal:** Reach a hand value closer to 21 than the dealer without going bust

---

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

---

## 📊 Report

The report is written **inside** `notebook.ipynb` and covers:

- **Introduction** — environment description, agent objective, research question
- **Dynamic Programming** — algorithm descriptions, parameter sensitivity (γ, θ), policy plots
- **Monte Carlo** — prediction and control results, hyperparameter tuning
- **Temporal Difference** — SARSA vs Q-Learning comparison, ε decay strategy
- **Comparison & Discussion** — MC vs TD metrics (cumulative reward, RMSE, sample efficiency)
- **Conclusion** — key findings and algorithm trade-offs

*Target length: ~1000–1500 words.*

---

## ⚙️ Setup

```bash
# Clone the repository
git clone <repo-url>
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

---

## 📅 Deadlines

| Date | Milestone |
|---|---|
| 09 Feb 2026 | Workgroup 1 — Environment setup |
| 20 Feb 2026 | Workgroup 2 — Dynamic Programming |
| 23 Feb 2026 | Workgroup 3 — Monte Carlo |
| 02 Mar 2026 | Workgroup 4 — Temporal Difference |
| 09 Mar 2026 | Workgroup 5 — Overflow / comparison |
| **05 Apr 2026** | **Final submission deadline (23:59)** |

---

## 📋 Grading Overview

| Component | Points |
|---|---|
| Dynamic Programming (code + report) | 2 pt |
| Monte Carlo (code + report) | 2 pt |
| Temporal Difference (code + report) | 2 pt |
| Report (intro, results, discussion, style) | 2 pt |
| Bonus (deep RL, exceptional environment) | +1 pt |

> Note: Predefined Gymnasium environments are not eligible for environment originality/correctness points. Max grade is capped at 10.0.

---

## 👥 Group

Made with ♠️ by Group **[X]** — Radboud University, Period 3, 2025–2026