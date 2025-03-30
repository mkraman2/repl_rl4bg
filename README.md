# Deep Reinforcement Learning for Closed-Loop Blood Glucose Control

This project is a simplified replication of the paper:
> *"Deep Reinforcement Learning for Closed-Loop Blood Glucose Control"*

It implements a Soft Actor-Critic (SAC) agent trained in a simulation environment powered by the UVA/Padova simulator (via `simglucose`). The agent learns to control blood glucose levels in Type-1 Diabetes patients using insulin injections.

---

## 🧠 Model Architecture

The agent is built on the **Soft Actor-Critic (SAC)** algorithm, using a partially observable Markov decision process (POMDP) formulation. It consists of:

### 🎮 Environment:
- `BloodGlucoseEnv`: Wraps the `T1DSimEnv` from `simglucose`, using a custom reward function based on the Magni risk index.
- `POMDPWrapper`: Augments the environment by stacking a history of past observations (temporal window) to deal with partial observability.

### 🤖 Agent (SAC):
- **Actor network**: MLP that maps observation history to insulin action (continuous).
- **Critic networks**: Two Q-value networks (Q1, Q2) and their targets (target Q1, Q2) used for double Q-learning.
- **Replay buffer**: Stores past transitions for off-policy training.

---

## 🚀 How to Run

### 1. Install Requirements
```bash
conda create -n glucose_rl python=3.12
conda activate glucose_rl
pip install -r requirements.txt
```
> Note: `simglucose` requires `gym==0.9.4` and may conflict with `gymnasium`. Use `gymnasium` only if using a compatible version of `simglucose`.

### 2. Run Training
```bash
python main.py --lr 3e-4 --alpha 0.2 --history-length 10 --episodes 1000
```

### 3. Optional Arguments
| Argument         | Description                            | Default  |
|------------------|----------------------------------------|----------|
| `--lr`           | Learning rate                          | `3e-4`   |
| `--alpha`        | Entropy temperature (exploration)      | `0.2`    |
| `--history-length` | Number of past observations to stack | `10`     |
| `--episodes`     | Number of training episodes            | `1000`   |

---

## 📊 Reward Function (Magni Risk Index)
The reward is computed from CGM readings using the **Magni risk function**, penalizing both hypoglycemia and hyperglycemia. It ensures safety-focused learning.

---

## 📁 File Structure
```
.
├── main.py           # Entry point: training loop
├── sac.py            # SAC agent implementation
├── environment.py    # Glucose simulator wrapper
├── pomdp.py          # POMDP history stacking wrapper
├── magni.py          # Magni risk-based reward function
└── requirements.txt  # Python dependencies
```

---

## 📈 Future Work
- Add glucose/insulin trend visualization
- Compare with PID or basal-bolus controllers
- Implement stochastic policy and log-prob entropy
- Extend to multi-day training with varying scenarios

---

## 🔬 Citation
If you use this repo or its ideas, please cite the original paper:
> Fox, Isaac, et al. "Deep reinforcement learning for closed-loop blood glucose control." *Nature Machine Intelligence* (2020).

---

## 🧪 Contributions Welcome!
Feel free to open issues, suggest features, or collaborate on expanding this prototype.

---

