# rl_bolts 
> rl_bolts is intended to be a package of nuts and bolts of RL algorithms, along with some full implementations of RL algorithms. 


rl_bolts is starting as a package of just nuts and bolts of RL, and algorithms (and new nuts and bolts) will be added over time, based on necessity.

## Install

`git clone https://github.com/jfpettit/rl_bolts.git`

`cd rl_bolts`

`pip install -e .`

## How to use

Import the bits you need to use in your code.

The bit below sets up an actor-critic network for the CartPole-v1 gym environment.

```python
import rl_bolts.neuralnets as nns
import gym
import torch
```

```python
env = gym.make("CartPole-v1")
actor_critic = nns.ActorCritic(
    env.observation_space.shape[0],
    env.action_space
)
```

We can print out the architecture of our actor_critic net below:

```python
actor_critic
```




    ActorCritic(
      (policy): CategoricalPolicy(
        (net): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=4, out_features=32, bias=True)
            (1): Linear(in_features=32, out_features=32, bias=True)
            (2): Linear(in_features=32, out_features=2, bias=True)
          )
        )
      )
      (value_f): MLP(
        (layers): ModuleList(
          (0): Linear(in_features=4, out_features=32, bias=True)
          (1): Linear(in_features=32, out_features=32, bias=True)
          (2): Linear(in_features=32, out_features=1, bias=True)
        )
      )
    )



```python
obs = env.reset()
action, logp, value = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
```

The cell above starts the environment in a new episode, and passes it through the actor-critic to get an action, action log probability, and value estimate for the state.

```python
print("action", action)
print("logp", logp)
print("value", value)
```

    action 1
    logp -0.8437294
    value 0.09548707

