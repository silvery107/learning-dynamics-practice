"""
Gym style Pendulum environment in Mujoco for data collection and visualization.
"""

import mujoco
import numpy as np
from gymnasium import spaces


class PendulumEnv:
    def __init__(self, model_path="assets/pendulum_model.xml"):
        """
        Initializes the Pendulum environment with the given Mujoco model.

        Args:
            model_path (str): Path to the Mujoco XML model file.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=self.model.actuator_ctrlrange[:, 0],
                                       high=self.model.actuator_ctrlrange[:, 1],
                                       dtype=np.float64)
        
        self.episode_length = 200
        self.current_step = 0
        self.reset()

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            np.array: The initial observation of the environment.
        """
        self.current_step = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)

        mujoco.mj_step(self.model, self.data)
        return self.state

    def step(self, action):
        """
        Takes a step in the environment using the provided action.

        Args:
            action (np.array): The action to take.
        Returns:
            tuple: A tuple containing (observation, reward, done, info).
        """
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        observation = self.state
        self.current_step += 1
        reward = None
        done = self.current_step >= self.episode_length
        info = {}
        return observation, reward, done, info

    @property
    def state(self):
        sensor_data = self.data.sensordata # [joint_pos, joint_vel]
        state = sensor_data.copy()
        return state
