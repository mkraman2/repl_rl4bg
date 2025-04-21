# Deep Reinforcement Learning for Closed-Loop Blood Glucose Control

This project is a simplified replication of the paper:
> **"Deep Reinforcement Learning for Closed-Loop Blood Glucose Control"**

It implements a Soft Actor-Critic (SAC) agent trained in a simulated environment powered by the UVA/Padova simulator (`simglucose`). The agent learns to control blood glucose in Type-1 Diabetes patients using insulin dosing decisions.

---

## 🧐 Model Architecture

The framework is based on the **Soft Actor-Critic (SAC)** algorithm, extended with support for:
- A **Transformer-based actor** ("sac-t") to learn temporal dependencies more effectively
- A baseline **PID controller** for comparison

### 🎮 Environment:
- `BloodGlucoseEnv`: Wraps the `T1DSimEnv` from `simglucose`, using a custom reward function based on the Magni risk index
- `POMDPWrapper`: Provides partial observability by stacking past `history_length` observations

### 🧠 Agents:
- **SAC (MLP)**: Classic SAC agent using a feedforward actor network
- **SAC-T (Transformer)**: Uses a Transformer encoder as the policy network for handling observation sequences
- **PID Controller**: Heuristic control based on proportional-integral-derivative tuning

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
conda create -n glucose_rl python=3.12
conda activate glucose_rl
pip install torch numpy matplotlib cloudpickle scipy pandas typing-extensions gym==0.26.2 gymnasium>=0.29
pip install --no-deps simglucose==0.2.11
```

### 2. Run Training
```bash
python main.py --controller sac
```

### 3. Available Arguments
| Argument             | Description                                      | Default  |
|----------------------|--------------------------------------------------|----------|
| `--lr`               | Learning rate                                    | `3e-4`   |
| `--alpha`            | Entropy temperature (exploration)                | `0.2`    |
| `--history-length`   | Number of past observations to stack             | `10`     |
| `--episodes`         | Number of training episodes                      | `1000`   |
| `--controller`       | Controller type: `sac`, `sac-t`, or `pid`        | `sac`    |

#### Example (Transformer Agent):
```bash
python main.py --lr 5e-3 --alpha 1.2 --history-length 40 --episodes 200 --controller "sac-t"
```

#### Example (PID Controller):
```bash
python main.py --controller pid
```

---

## 🔢 Reward Function

The reward is calculated from CGM values using the **Magni Risk Index**, which penalizes both hypoglycemia and hyperglycemia:

```python
reward = -magni_risk(glucose)
```

This formulation ensures the controller learns to maintain glucose in a safe range.

---

## 📈 Logs and Visualization

After training, a log file `glucose_log.csv` is created. You can visualize training performance with:
```bash
python main.py  # logs automatically plotted at end
```

The plot shows:
- CGM range (min/mean/max) per episode
- Total reward per episode
- Average insulin dosed per episode
- CGM and insulin profile for the best-performing episode

---

## 📁 File Structure
```
.
├── main.py              # Main entry point and training loop
├── sac.py               # SAC agent and replay buffer
├── environment.py       # Glucose simulator wrapper
├── pomdp.py             # Observation wrapper with history stacking
├── pid_controller.py    # PID controller logic
├── magni.py             # Risk-based reward computation
├── logger.py            # Logging utility
├── plot_logs.py         # Visualization script
├── requirements.txt     # Dependencies
└── README.md
```

---

## 📊 Future Work
- Support multi-patient training
- Add stochastic policy with log-prob entropy
- Incorporate meal/CHO events as inputs
- Tune Transformer depth/heads for larger history windows

---

## 🎓 Citation
If you use this work or ideas, please cite the original paper:
> Fox, Isaac, et al. "Deep reinforcement learning for closed-loop blood glucose control." *Nature Machine Intelligence* (2020).

---

## 🚜 Contributions Welcome
Pull requests, issues, and improvements are welcome! Help us explore safer RL in healthcare settings.

