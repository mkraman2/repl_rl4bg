import pandas as pd
import matplotlib.pyplot as plt

def plot_from_log(log_path="glucose_log.csv", history_length=10):  # 👈 Add history_length as argument
    df = pd.read_csv(log_path)

    # New: If 'env_id' doesn't exist (old CSVs), create a dummy one
    if "env_id" not in df.columns:
        df["env_id"] = 0

    # Group by both (episode, env_id)
    grouped = df.groupby(["episode", "env_id"])

    avg_glucose = grouped["glucose"].mean()
    min_glucose = grouped["glucose"].min()
    max_glucose = grouped["glucose"].max()
    sum_reward = grouped["reward"].sum()
    avg_reward_per_step = grouped["reward"].sum() / grouped["reward"].count()
    avg_insulin = grouped["insulin"].mean()

    # Find (episode, env_id) pair with best reward
    best_idx = avg_reward_per_step.idxmax()
    best_episode, best_env = best_idx

    best_data = df[(df["episode"] == best_episode) & (df["env_id"] == best_env)]

    # === NEW ===
    # Skip the warmup history_length timesteps
    best_data = best_data[best_data["timestep"] >= history_length]

    timesteps = best_data["timestep"]
    glucose = best_data["glucose"]
    insulin = best_data["insulin"]

    plt.figure(figsize=(20, 5))

    # Glucose Range per Episode (averaged over envs)
    plt.subplot(1, 4, 1)
    avg_glucose_by_episode = df.groupby("episode")["glucose"].mean()
    min_glucose_by_episode = df.groupby("episode")["glucose"].min()
    max_glucose_by_episode = df.groupby("episode")["glucose"].max()
    plt.plot(avg_glucose_by_episode, label="Mean CGM", marker='o')
    plt.plot(min_glucose_by_episode, label="Min CGM", linestyle='--', color='orange')
    plt.plot(max_glucose_by_episode, label="Max CGM", linestyle='--', color='green')
    plt.title("Glucose Range per Episode")
    plt.xlabel("Episode")
    plt.ylabel("CGM (mg/dL)")
    plt.legend()

    # Total Reward per Episode
    plt.subplot(1, 4, 2)
    sum_reward_by_episode = df.groupby("episode")["reward"].sum()
    plt.plot(sum_reward_by_episode, marker='x', color='red')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Average Insulin per Episode
    plt.subplot(1, 4, 3)
    avg_insulin_by_episode = df.groupby("episode")["insulin"].mean()
    plt.plot(avg_insulin_by_episode, marker='s', color='purple')
    plt.title("Average Insulin Dose per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Insulin (U)")

    # Best CGM and Insulin
    plt.subplot(1, 4, 4)
    ax1 = plt.gca()
    ax1.plot(timesteps, glucose, color='tab:blue', label='CGM')
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("CGM (mg/dL)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, insulin, color='tab:purple', linestyle='--', label='Insulin Dose')
    ax2.set_ylabel("Insulin (U)", color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    plt.title(f"Best Episode {best_episode}, Env {best_env} - CGM & Insulin (No Warmup)")
    plt.tight_layout()
    plt.show()
