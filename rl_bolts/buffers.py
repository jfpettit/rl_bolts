# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_buffers.ipynb (unless otherwise specified).

__all__ = ['PGBuffer', 'ReplayBuffer']

# Cell
import numpy as np
import scipy
from typing import Optional, Any, Union
import torch

# Cell
class PGBuffer:
    """
    A buffer for storing trajectories experienced by an agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    This class was written by Joshua Achaim at OpenAI.

    Args:
    - obs_dim (tuple or int): Dimensionality of input feature space.
    - act_dim (tuple or int): Dimensionality of action space.
    - size (int): buffer size.
    - gamma (float): reward discount factor.
    - lam (float): Lambda parameter for GAE-Lambda advantage estimation
    """
    def __init__(
        self,
        obs_dim: Union[tuple, int],
        act_dim: Union[tuple, int],
        size: int,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.95,
    ):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.array,
        act: np.array,
        rew: Union[int, float, np.array],
        val: Union[int, float, np.array],
        logp: Union[float, np.array],
    ):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Args:
        - obs (np.array): Current observation to store.
        - act (np.array): Current action.
        - rew (int or float or np.array): Current reward from environment.
        - val (int or float or np.array): Value estimate for the current state.
        - logp (float or np.array): log probability of chosen action under current policy distribution.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: Optional[Union[int, float, np.array]] = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).

        Args:
        - last_val (int or float or np.array): Estimate of rewards-to-go. If trajectory ended, is 0.
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def _combined_shape(
        self, length: Union[int, np.array], shape: Optional[Union[int, tuple]] = None
    ):
        """
        Return tuple of combined shapes from input length and tuple describing shape.

        Args:
        - length (int or np.array): Length of resultant shape.
        - shape (int or tuple): Other shape dimensions to combine.

        Returns:
        - tuple of shape dimensions
        """
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x: np.array, discount: float):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# Cell

class ReplayBuffer(PGBuffer):
    """
    A replay buffer for off-policy RL agents.

    This class is borrowed from OpenAI's SpinningUp package: https://spinningup.openai.com/en/latest/

    Args:
    - obs_dim (tuple or int): Dimensionality of input feature space.
    - act_dim (tuple or int): Dimensionality of action space.
    - size (int): buffer size.
    """

    def __init__(
        self, obs_dim: Union[tuple, int], act_dim: Union[tuple, int], size: int
    ):
        self.obs1_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.array,
        act: Union[float, int, np.array],
        rew: Union[float, int],
        next_obs: np.array,
        done: bool,
    ):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Args:
        - obs (np.array): Current observations.
        - act (float or int or np.array): Current action.
        - rew (float or int): Current reward
        - next_obs (np.array): Observations from next environment step.
        - done (bool): Whether the episode has reached a terminal state.
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: Optional[int] = 32):
        """
        Sample a batch of agent-environment interaction from the buffer.

        Args:
        - batch_size (int): Number of interactions to sample for the batch.

        Returns:
        - tuple of batch tensors
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return tuple(torch.as_tensor(v, dtype=torch.float32) for _, v in batch.items())

    def get(self):
        """
        Get all contents of the batch.

        Returns:
        - list of numpy arrays; full contents of the buffer.
        """
        return [self.obs1_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf]