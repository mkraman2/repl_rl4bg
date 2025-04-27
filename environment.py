# environment.py

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.risk import risk_index
from simglucose.controller.base import Action
from datetime import datetime, timedelta
from magni import reward_from_risk

class BloodGlucoseEnv(gym.Env):
    def __init__(self, patient_name="adult#001", seed=None):
        super().__init__()
        self.patient_name = patient_name
        self.seed_value = seed
        self._build_sim()

    def _build_sim(self):
        now = datetime.now()
        random_hour = np.random.randint(0, 24)  # Randomize start hour
        start_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=random_hour)

        patient = T1DPatient.withName(self.patient_name)
        scenario = RandomScenario(start_time=start_time)
        sensor = CGMSensor.withName('Dexcom', seed=self.seed_value)
        pump = InsulinPump.withName('Insulet')

        self.env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

        self.last_insulin = 0.0
        self.last_meal = 0.0

        self.observation_space = gym.spaces.Box(
            low=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([600.0, 0.5, 100.0], dtype=np.float32)
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([0.5]), dtype=np.float32)

    def reset(self, dummy_length=20):
        self.env.reset()

        dummy_action = Action(basal=0.0, bolus=0.0)

        step = None
        for step_idx in range(dummy_length):
            step = self.env.step(dummy_action)

        self.last_insulin = 0.0
        self.last_meal = getattr(step, "meal", 0.0)

        # Only after stepping, print CGM
        # print(f"[DEBUG] After dummy steps: CGM={step.observation.CGM}")

        return self._process_obs(step.observation), {}


    def step(self, action):
        action_value = float(np.clip(action[0], 0.0, 0.5))
        action_obj = Action(basal=0.0, bolus=action_value)

        step = self.env.step(action_obj)
        obs = step.observation
        reward = reward_from_risk(obs.CGM)

        self.last_insulin = getattr(step, "insulin", action_value)
        self.last_meal = getattr(step, "meal", 0.0)

        info = {
            "CGM": obs.CGM,
            "insulin": self.last_insulin,
            "meal": self.last_meal,
            "bg": getattr(step, "bg", None),
            "risk": getattr(step, "risk", None),
        }

        terminated = step.done
        truncated = False

        return self._process_obs(obs), reward, terminated, truncated, info

    def _process_obs(self, obs):
        return np.array([obs.CGM, self.last_insulin, self.last_meal], dtype=np.float32)
