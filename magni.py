# magni.py
import numpy as np

def magni_risk(glucose):
    """
    Compute the Magni risk index based on blood glucose value.
    :param glucose: blood glucose level (mg/dL)
    :return: risk index (higher is worse)
    """
    glucose = np.clip(glucose, 1e-3, None)  # avoid log(0)
    f = 1.509 * ((np.log(glucose)) ** 1.084 - 5.381)
    risk = np.where(f < 0, -1, 1) * 10 * (f ** 2)
    return risk


def reward_from_risk(glucose):
    """
    Reward is negative risk. Safer glucose = higher reward.
    """
    return -np.abs(magni_risk(glucose))  # Penalize both high and low
