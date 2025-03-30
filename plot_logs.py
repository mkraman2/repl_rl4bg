# plot_logs.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_from_log(log_path="glucose_log.csv"):
    df = pd.read_csv(log_path)

    grouped = df.groupby("episode")
    avg_glucose = grouped["glucose"].mean()
    min_glucose = grouped["glucose"].min()
    max_glucose = grouped["glucose"].max()
    sum_reward = grouped["reward"].sum()
    avg_reward_per_step = grouped["reward"].sum() / grouped["reward"].count()
    avg_insulin = grouped["insulin"].mean()

    best_episode = avg_reward_per_step.idxmax()
    best_data = df[df["episode"] == best_episode]
    timesteps = best_data["timestep"]
    glucose = best_data["glucose"]
    insulin = best_data["insulin"]

    plt.figure(figsize=(20, 5))

    # Glucose plot
    plt.subplot(1, 4, 1)
    plt.plot(avg_glucose, label="Mean CGM", marker='o')
    plt.plot(min_glucose, label="Min CGM", linestyle='--', color='orange')
    plt.plot(max_glucose, label="Max CGM", linestyle='--', color='green')
    plt.title("Glucose Range per Episode")
    plt.xlabel("Episode")
    plt.ylabel("CGM (mg/dL)")
    plt.legend()

    # Reward plot (total reward per episode)
    plt.subplot(1, 4, 2)
    plt.plot(sum_reward, marker='x', color='red')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Insulin plot
    plt.subplot(1, 4, 3)
    plt.plot(avg_insulin, marker='s', color='purple')
    plt.title("Average Insulin Dose per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Insulin (U)")

    # Best episode CGM and insulin
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

    plt.title(f"Best Episode ({best_episode}) - CGM & Insulin")
    plt.tight_layout()
    plt.show()