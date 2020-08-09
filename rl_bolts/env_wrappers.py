# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_env_wrappers.ipynb (unless otherwise specified).

__all__ = ['ToTorchWrapper', 'StateNormalizeWrapper', 'RewardScalerWrapper', 'BestPracticesWrapper']

# Cell
import gym
import numpy as np
import torch
from typing import Optional, Union

# Cell
class ToTorchWrapper(gym.Wrapper):
    """
    Environment wrapper for converting actions from torch.Tensors to np.array and converting observations from np.array to
    torch.Tensors.

    Args:
    - env (gym.Env): Environment to wrap. Should be a subclass of gym.Env and follow the OpenAI Gym API.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.env = env

    def reset(self, *args, **kwargs):
        """
        Reset the environment.

        Returns:
        - tensor_obs (torch.Tensor): output of reset as PyTorch Tensor.
        """
        obs = self.env.reset(*args, **kwargs)
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
        return tensor_obs

    def step(self, action: torch.Tensor, *args, **kwargs):
        """
        Execute environment step.

        Converts from torch.Tensor action and returns observations as a torch.Tensor.

        Returns:
        - tensor_obs (torch.Tensor): Next observations as pytorch tensor.
        - reward (float or int): The reward earned at the current timestep.
        - done (bool): Whether the episode is in a terminal state.
        - infos (dict): The info dict from the environment.
        """

        action = self.action2np(action)
        obs, reward, done, infos = self.env.step(action, *args, **kwargs)
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
        return tensor_obs, reward, done, infos

    def action2np(self, action: torch.Tensor):
        """
        Convert torch.Tensor action to NumPy.

        Args:
        - action (torch.Tensor): The action to convert.

        Returns:
        - np_act (np.array or int): The action converted to numpy.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_map = lambda action: int(action.squeeze().numpy())
        if isinstance(self.action_space, gym.spaces.Box):
            action_map = lambda action: action.numpy()

        np_act = action_map(action)
        return np_act

# Cell
class StateNormalizeWrapper(gym.Wrapper):
    """
    Environment wrapper for normalizing states.

    Args:
    - env (gym.Env): Environment to wrap.
    - beta (float): Beta parameter for running mean and variance calculation.
    - eps (float): Parameter to avoid division by zero in case variance goes to zero.
    """
    def __init__(self, env: gym.Env, beta: Optional[float] = 0.99, eps: Optional[float] = 1e-8):
        super().__init__(env)

        self.env = env

        self.mean = np.zeros(self.observation_space.shape)
        self.var = np.ones(self.observation_space.shape)

        self.beta = beta
        self.eps = eps

    def normalize(self, state: np.array):
        """
        Update running mean and variance parameters and normalize input state.

        Args:
        - state (np.array): State to normalize and to use to calculate update.

        Returns:
        - norm_state (np.array): Normalized state.
        """
        self.mean = self.beta * self.mean + (1. - self.beta) * state
        self.var = self.beta * self.var + (1. - self.beta) * np.square(state - self.mean)
        norm_state = (state - self.mean) / (np.sqrt(self.var) + self.eps)
        return norm_state

    def reset(self, *args, **kwargs):
        """
        Reset environment and return normalized state.

        Returns:
        - norm_state (np.array): Normalized state.
        """
        state = self.env.reset()
        norm_state = self.normalize(state)
        return norm_state

    def step(self, action: Union[np.array, int, float], *args, **kwargs):
        """
        Step environment and normalize state.

        Args:
        - action (np.array or int or float): Action to use to step the environment.

        Returns:
        - norm_state (np.array): Normalized state.
        - reward (int or float): Reward earned at step.
        - done (bool): Whether the episode is over.
        - infos (dict): Any infos from the environment.
        """
        state, reward, done, infos = self.env.step(action, *args, **kwargs)
        norm_state = self.normalize(state)
        return norm_state, reward, done, infos

# Cell
class RewardScalerWrapper(gym.Wrapper):
    """
    A class for reward scaling over training.

    Calculates running mean and standard deviation of observed rewards and scales the rewards using the variance.

    Computes: $r_t / (\sigma + eps)$
    """
    def __init__(self, env: gym.Env, beta: Optional[float] = 0.99, eps: Optional[float] = 1e-8):
        super().__init__(env)

        self.beta = beta
        self.eps = eps

        self.var = 1
        self.mean = 0

    def scale(self, reward: Union[int, float]):
        """
        Update running mean and variance for rewards, scale reward using the variance.

        Args:
        - reward (int or float): reward to scale.

        Returns:
        - scaled_rew (float): reward scaled using variance.
        """
        self.mean = self.beta * self.mean + (1. - self.beta) * reward
        self.var = self.beta * self.var + (1. - self.beta) * np.square(reward - self.mean)

        scaled_rew = reward / (np.sqrt(self.var) + self.eps)

        return scaled_rew

    def step(self, action, *args, **kwargs):
        """
        Step the environment and scale the reward.

        Args:
        - action (np.array or int or float): Action to use to step the environment.

        Returns:
        - state (np.array): Next state from environment.
        - scaled_rew (float): reward scaled using the variance.
        - done (bool): Indicates whether the episode is over.
        - infos (dict): Any information from the environment.
        """
        state, reward, done, infos = self.env.step(action, *args, **kwargs)
        scaled_rew = self.scale(reward)
        return state, scaled_rew, done, infos

# Cell
class BestPracticesWrapper(gym.Wrapper):
    """
    This wrapper combines the wrappers which we think (from experience and from reading papers/blogs and watching lectures)
    constitute best practices.

    At the moment it combines the wrappers below in the order listed:
    1. StateNormalizeWrapper
    2. RewardScalerWrapper
    3. ToTorchWrapper

    Args:
    - env (gym.Env): Environment to wrap.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        env = StateNormalizeWrapper(env)
        env = RewardScalerWrapper(env)
        self.env = ToTorchWrapper(env)

    def reset(self):
        """
        Reset environment.

        Returns:
        - obs (torch.Tensor): Starting observations from the environment.
        """
        obs = self.env.reset()
        return obs

    def step(self, action, *args, **kwargs):
        """
        Step the environment forward using input action.

        Args:
        - action (torch.Tensor): Action to step the environment with.

        Returns:
        - obs (torch.Tensor): Next step observations.
        - reward (int or float): Reward for the last timestep.
        - done (bool): Whether the episode is over.
        - infos (dict): Dictionary of any info from the environment.
        """
        obs, reward, done, infos = self.env.step(action, *args, **kwargs)
        return obs, reward, done, infos