# environment.py

import gymnasium as gym
import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.risk import risk_index
from simglucose.controller.base import Action
from datetime import datetime
from magni import reward_from_risk

class BloodGlucoseEnv(gym.Env):
    def __init__(self):
        super(BloodGlucoseEnv, self).__init__()

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())        
        patient = T1DPatient.withName('adult#001')
        scenario = RandomScenario(start_time=start_time)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')

        self.last_insulin = 0.0
        self.last_meal = 0.0

        self.env = T1DSimEnv(
            patient=patient,
            sensor=sensor,
            pump=pump,
            scenario=scenario,
        )

        # Define Gym-compatible observation and action spaces manually
        self.observation_space = gym.spaces.Box(
            low=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([600.0, 0.5, 100.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([0.5]),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        self.last_insulin = 0.0
        self.last_meal = 0.0
        return self._process_obs(obs.observation), {}

    def step(self, action):
        # Clip and wrap action into simglucose's expected format
        action_value = float(np.clip(action[0], 0.0, 0.5))
        action_obj = Action(basal=0.0, bolus=action_value)

        # Step through the simglucose environment
        step = self.env.step(action_obj)

        # Extract info from the step safely
        obs = step.observation
        glucose = obs.CGM
        reward = reward_from_risk(glucose)

        # Fallback if attributes are missing
        self.last_insulin = getattr(step, "insulin", action_value)
        self.last_meal = getattr(step, "meal", 0.0)

        info = {
            "CGM": glucose,
            "insulin": self.last_insulin,
            "meal": self.last_meal,
            "bg": getattr(step, "bg", None),
            "risk": getattr(step, "risk", None),
        }

        terminated = step.done
        truncated = False

        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        return np.array([
            obs.CGM,
            self.last_insulin,
            self.last_meal
        ], dtype=np.float32)