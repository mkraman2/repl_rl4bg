import csv
import os

class GlucoseLogger:
    def __init__(self, log_path="glucose_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'timestep', 'glucose', 'insulin', 'reward'])

    def log_step(self, episode, timestep, glucose, insulin, reward):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, timestep, glucose, insulin, reward])
