# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_neuralnets.ipynb (unless otherwise specified).

__all__ = ['MLP', 'Actor', 'CategoricalPolicy', 'GaussianPolicy', 'ActorCritic', 'MLPQActor', 'MLPQFunction']

# Cell
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import gym
from scipy.signal import lfilter
from typing import Optional, Iterable, List, Dict, Callable, Union, Tuple
from .env_wrappers import ToTorchWrapper

# Cell
class MLP(nn.Module):
    r"""
    A class for building a simple MLP network.

    Args:
    - layer_sizes (list or tuple): Layer sizes for the network.
    - activations (Function): Activation function for MLP net.
    - out_act (Function): Output activation function
    - out_squeeze (bool): Whether to squeeze the output of the network.
    """

    def __init__(
        self,
        layer_sizes: Union[List, Tuple],
        activations: Optional[Callable] = torch.tanh,
        out_act: Optional[bool] = None,
        out_squeeze: Optional[bool] = False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations
        self.out_act = out_act
        self.out_squeeze = out_squeeze

        for i, l in enumerate(layer_sizes[1:]):
            self.layers.append(nn.Linear(layer_sizes[i], l))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP network"""
        for l in self.layers[:-1]:
            x = self.activations(l(x))

        if self.out_act is None:
            x = self.layers[-1](x)
        else:
            x = self.out_act(self.layers[-1](x))

        return torch.squeeze(x, -1) if self.out_squeeze else x

# Cell
class Actor(nn.Module):
    """
    Barebones class structure for an Actor.
    """
    def action_distribution(self, states):
        raise NotImplementedError

    def logprob_from_distribution(self, policy, action):
        raise NotImplementedError

    def forward(self, x, a = None):
        """
        Forward pass for an policy.

        Args:
        - x (torch.Tensor): Input state from the environment.
        - a (torch.Tensor): Action that was taken.

        Returns:
        - policy (PyTorch distribution): The policy distribution.
        - logp_a (torch.Tensor): Log-probability of input action under the policy distribution.
        """
        policy = self.action_distribution(x)
        logp_a = None
        if a is not None:
            logp_a = self.logprob_from_distribution(policy, a)
        return policy, logp_a

# Cell
class CategoricalPolicy(Actor):
    r"""
    A class for a Categorical Policy network. Used in discrete action space environments.

    The policy is an `MLP`.

    Args:
    - state_features (int): Dimensionality of the state space.
    - action_dim (int): Dimensionality of the action space.
    - hidden_sizes (list or tuple): Hidden layer sizes.
    - activation (Function): Activation function for the network.
    - out_activation (Function): Output activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
    ):
        super().__init__()
        self.net = MLP(
            [state_features] + list(hidden_sizes) + [action_dim], activations=activation
        )

    def action_distribution(self, x: torch.Tensor):
        """
        Defines action distribution conditioned on input state.

        Args:
        - x(torch.Tensor): input state

        Returns:
        - Categorical distribution: Policy over the action space.
        """
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    def logprob_from_distribution(self, policy: torch.distributions.Distribution, actions: torch.Tensor):
        """
        Calculate the log-probability of an action under a policy.

        Args:
        - policy (torch.distributions.Distribution): The policy distribution over input state.
        - actions (torch.Tensor): Actions to take log probability of.

        Returns:
        - log_probs (torch.Tensor): Log-probabilities of actions under the policy distribution.
        """
        return policy.log_prob(actions)

# Cell
class GaussianPolicy(Actor):
    r"""
    A class for a Gaussian Policy network. Used in continuous action space environments. The policy is an `MLP`.

    Args:
    - state_features (int): Dimensionality of the state space.
    - action_dim (int): Dimensionality of the action space.
    - hidden_sizes (list or tuple): Hidden layer sizes.
    - activation (Function): Activation function for the network.
    - out_activation (Function): Output activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
    ):
        super().__init__()

        self.net = MLP(
            [state_features] + list(hidden_sizes) + [action_dim],
            activations=activation,
            out_act=out_activation,
        )

        self.logstd = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def action_distribution(self, states):
        """
        Defines action distribution conditioned on input state.

        Args:
        - x(torch.Tensor): input state

        Returns:
        - Normal distribution: Policy over the action space.
        """
        mus = self.net(states)
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(mus, std)

    def logprob_from_distribution(self, policy, actions):
        """
        Calculate the log-probability of an action under a policy.

        Args:
        - policy (torch.distributions.Distribution): The policy distribution over input state.
        - actions (torch.Tensor): Actions to take log probability of.

        Returns:
        - log_probs (torch.Tensor): Log-probabilities of actions under the policy distribution.
        """
        return policy.log_prob(actions).sum(axis=-1)

# Cell
class ActorCritic(nn.Module):
    r"""
    An Actor Critic class for Policy Gradient algorithms.

    Has built-in capability to work with continuous (gym.spaces.Box) and discrete (gym.spaces.Discrete) action spaces.
    The policy and value function are both `MLP`.

    If working with a different action space,
    the user can pass in a custom policy class for that action space as an argument.

    Args:
    - state_features (int): Dimensionality of the state space.
    - action_space (gym.spaces.Space): Action space of the environment.
    - hidden_sizes (list or tuple): Hidden layer sizes.
    - activation (Function): Activation function for the network.
    - out_activation (Function): Output activation function for the network.
    - policy (nn.Module): Custom policy class for an environment where the action space is not gym.spaces.Box or gym.spaces.Discrete

    """

    def __init__(
        self,
        state_features: int,
        action_space: int,
        hidden_sizes: Optional[Union[Tuple, List]] = (32, 32),
        activation: Optional[Callable] = torch.tanh,
        out_activation: Optional[Callable] = None,
        policy: Optional[nn.Module] = None,
    ):
        super(ActorCritic, self).__init__()

        obs_dim = state_features

        if isinstance(action_space, gym.spaces.Discrete):
            act_dim = action_space.n
            pol = CategoricalPolicy

        elif isinstance(action_space, gym.spaces.Box):
            act_dim = action_space.shape[0]
            pol = GaussianPolicy
        else:
            act_dim = action_space
            pol = policy

        self.policy = pol(
            obs_dim,
            act_dim,
            hidden_sizes,
            activation,
            out_activation
        )

        self.value_f = MLP(
            [state_features] + list(hidden_sizes) + [1],
            activations=activation,
            out_squeeze=True,
        )

    def step(self, x: torch.Tensor):
        """
        Get action, action log probability, and value estimate for an input state.

        Args:
        - x (torch.Tensor): input state.

        Returns:
        - action (torch.Tensor): Action chosen by the policy.
        - logp_action (torch.Tensor): Log probability of that action chosen by the policy.
        - value (torch.Tensor): Value estimate of the current state.
        """
        with torch.no_grad():
            policy = self.policy.action_distribution(x)
            action = policy.sample()
            logp_action = self.policy.logprob_from_distribution(policy, action)
            value = self.value_f(x)
        return action, logp_action, value

    def act(self, x: torch.Tensor):
        """
        Similar to `step`, but get only the action.

        Args:
        - x (torch.Tensor): input state

        Returns:
        - action (torch.Tensor): Action chosen by the policy.
        """
        return self.step(x)[0]

# Cell

class MLPQActor(nn.Module):
    r"""
    An actor for Q policy gradient algorithms.

    The policy is an `MLP`.
    The output from the policy network is scaled to action space limits on the forward pass.

    Args:
    - state_features (int): Dimensionality of the state space.
    - action_dim (int): Dimensionality of the action space.
    - hidden_sizes (list or tuple): Hidden layer sizes.
    - activation (Function): Activation function for the network.
    - action_limit (float or int): Limits of the action space.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[list, tuple],
        activation: Callable,
        action_limit: Union[float, int],
    ):
        super(MLPQActor, self).__init__()
        policy_layer_sizes = [state_features] + list(hidden_sizes) + [action_dim]
        self.policy = MLP(policy_layer_sizes, activation, torch.tanh)
        self.action_limit = action_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return output from the policy network scaled to the limits of the env action space.
        Args:
        - x (torch.Tensor): States from environment.

        Returns:
        - scaled_action (torch.Tensor): Action scaled to action space limits.
        """
        scaled_action = self.action_limit * self.policy(x)
        return scaled_action

# Cell
class MLPQFunction(nn.Module):
    r"""
    A Q function network for Q policy gradient methods.

    The Q function is an `MLP`. It always takes in a (state, action) pair and returns a Q-value estimate for that pair.

    Args:
    - state_features (int): Dimensionality of the state space.
    - action_dim (int): Dimensionality of the action space.
    - hidden_sizes (list or tuple): Hidden layer sizes.
    - activation (Function): Activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[tuple, list],
        activation: Callable,
    ):
        super().__init__()
        self.qfunc = MLP(
            [state_features + action_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Return Q-value estimate for state, action pair (x, a).

        Args:
        - x (torch.Tensor): Environment state.
        - a (torch.Tensor): Action taken by the policy.

        Returns:
        - q (torch.Tensor): Q-value estimate for state action pair.
        """
        q = self.qfunc(torch.cat([x, a], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
