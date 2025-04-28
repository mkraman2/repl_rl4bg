# Deep Reinforcement Learning for Closed-Loop Blood Glucose Control

This project is a simplified replication of the paper:\
**"Deep Reinforcement Learning for Closed-Loop Blood Glucose Control"**

It implements a Soft Actor-Critic (SAC) agent trained in a simulated environment powered by the UVA/Padova simulator (`simglucose`). The agent learns to control blood glucose in Type-1 Diabetes patients using insulin dosing decisions.

---

## 😬 Model Architecture

The framework is based on the **Soft Actor-Critic (SAC)** algorithm, extended with support for:

- A **Transformer-based actor** ("sac-t") to learn temporal dependencies more effectively
- A baseline **PID controller** for comparison

### 🎮 Environment:

- `BloodGlucoseEnv`: Wraps the `T1DSimEnv` from `simglucose`, using a custom reward function based on the Magni risk index
- `POMDPWrapper`: Provides partial observability by stacking past `history_length` observations

### 🧀 Agents:

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

---

## 🛠️ Available Arguments

| Argument            | Description                                    | Default |
| ------------------- | ---------------------------------------------- | ------- |
| `--lr`              | Learning rate                                  | `3e-4`  |
| `--alpha`           | Entropy temperature (exploration)              | `0.2`   |
| `--history-length`  | Number of past observations to stack           | `10`    |
| `--episodes`        | Number of training episodes                    | `1000`  |
| `--controller`      | Controller type: `sac`, `sac-t`, or `pid`      | `sac`   |
| `--num-envs`        | Number of parallel environments                | `32`    |
| `--dnn-hidden-size` | Hidden layer size for MLP/Transformer networks | `128`   |

📋 `--num-envs` helps parallelize experience collection for faster learning\
📋 `--dnn-hidden-size` controls model capacity (larger = more expressive networks)

---

## 🎯 Example Commands

Train a standard SAC agent:

```bash
python main.py --lr 3e-4 --controller sac --episodes 500
```

Train a Transformer-based SAC agent:

```bash
python main.py --controller sac-t --history-length 20 --dnn-hidden-size 64 --episodes 500
```

Use multiple environments to accelerate experience collection:

```bash
python main.py --num-envs 64 --controller sac
```

Use a PID controller:

```bash
python main.py --controller pid
```

---

## 🔢 Reward Function

The reward is calculated from CGM values using the **Magni Risk Index**, which penalizes both hypoglycemia and hyperglycemia:

```python
reward = -magni_risk(glucose)
```

This ensures the controller learns to maintain glucose within a safe range.

---

## 📈 Logs and Visualization

After training, a log file `glucose_log.csv` is automatically created.

You can visualize training progress and policy performance with:

```bash
python main.py
```

The generated plot shows:

- CGM (min/mean/max) per episode
- Total reward per episode
- Average insulin dose per episode
- CGM and insulin profiles for the best-performing episode

---

## 📁 File Structure

```
.
├── main.py              # Main training script
├── sac.py               # SAC agent and replay buffer
├── environment.py       # Glucose simulator wrapper
├── pomdp.py             # Observation wrapper with history stacking
├── pid_controller.py    # PID controller baseline
├── magni.py             # Risk-based reward function
├── logger.py            # Training logger
├── plot_logs.py         # Training visualizations
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## 📊 Future Work

- Support multi-patient training and fine-tuning
- Add stochastic policies with entropy regularization
- Include meal/CHO events as explicit inputs
- Tune Transformer depth and number of heads for larger history windows

---

## 🎓 Citation

If you use this work or ideas, please cite the original paper:

> Fox, Isaac, et al. "Deep reinforcement learning for closed-loop blood glucose control." *Nature Machine Intelligence* (2020).

---

## 🚜 Contributions Welcome

Pull requests, issues, and improvements are welcome! Help us explore safer RL for healthcare applications.

