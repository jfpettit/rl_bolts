# rl_bolts 
> rl_bolts is intended to be a package of nuts and bolts of RL algorithms, along with some full implementations of RL algorithms. 


rl_bolts is starting as a package of just nuts and bolts of RL, and algorithms (and new nuts and bolts) will be added over time, based on necessity.

## Install

`git clone https://github.com/jfpettit/rl_bolts.git`

`cd rl_bolts`

`pip install -r requirements.txt`

## How to use

Import the bits you need to use in your code.

The bit below sets up an actor-critic network for the CartPole-v1 gym environment.

```
import rl_bolts.neuralnets as nns
import gym
import torch
```

```
env = gym.make("CartPole-v1")
actor_critic = nns.ActorCritic(
    env.observation_space.shape[0],
    env.action_space
)
```

We can print out the architecture of our actor_critic net below:

```
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



```
obs = env.reset()
action, logp, value = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
```

The cell above starts the environment in a new episode, and passes it through the actor-critic to get an action, action log probability, and value estimate for the state.

```
print("action", action)
print("logp", logp)
print("value", value)
```

    action tensor(0)
    logp tensor(-0.6733)
    value tensor(0.1155)


## Using a pre-built algorithm

While the primary aim of this package is to provide some building blocks for RL algorithms, we'll also provide implementations of a few plug-and-play algorithms. At present, we've implemented `PPO` (it still needs to be thoroughly benchmarked, so be aware of that). Here is how to use it.

```
from rl_bolts.algorithms import PPO # import the PPO algorithm
import pytorch_lightning as pl # PPO is a pytorch-lightning module, so need their library for Trainer.
env_to_train_in = "CartPole-v1" # set env to train PPO in. 
agent = PPO(env_to_train_in) # initialize agent
trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=1) # set up trainer, in practice you'd set max_epochs to more than one
trainer.fit(agent) # run trainer
```
