"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax                        # CHANGE 1
import jax.numpy as jnp
from flax import linen as nn

from continuous_control.networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:  # CHANGE 1
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jax.Array], jax.Array] = nn.relu  # CHANGE 1

    @nn.compact
    def __call__(self, observations: jax.Array,
                 actions: jax.Array) -> jax.Array:            # CHANGE 1
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jax.Array], jax.Array] = nn.relu  # CHANGE 1

    @nn.compact
    def __call__(self, observations: jax.Array,
                 actions: jax.Array) -> Tuple[jax.Array, jax.Array]:  # CHANGE 1
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2