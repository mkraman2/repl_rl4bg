# environment.py

import gymnasium as gym
import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.risk import risk_index
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

        self.env = T1DSimEnv(
            patient=patient,
            sensor=sensor,
            pump=pump,
            scenario=scenario,
        )

        # Define Gym-compatible observation and action spaces manually
        self.observation_space = gym.spaces.Box(
            low=np.array([40.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([400.0, 150.0, 15.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([10.0]),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs.observation), {}

    def step(self, action):
        # Map action (units of insulin) into simglucose controller interface
        action_value = float(np.clip(action[0], 0.0, 10.0))
        action_dict = {'bolus': action_value, 'basal': 0.0}

        step = self.env.step(action_dict)
        obs = step.observation
        done = step.done
        info = step.info
        glucose = obs.CGM
        reward = reward_from_risk(glucose)

        terminated = step.done
        truncated = False
        
        return self._process_obs(obs), reward, done, info
    
    def _process_obs(self, obs):
        # Return a float32 array of [CGM, insulin]
        return np.array([
            obs.CGM,
        ], dtype=np.float32)
