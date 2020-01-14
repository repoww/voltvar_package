from rllab.core.serializable import Serializable

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import Policy
from contextlib import contextmanager

import numpy as np


class GreedyPolicy(Policy, Serializable):
    """
    Fixed policy that randomly samples actions uniformly at random.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, env_spec, qf, epsilon):
        Serializable.quick_init(self, locals())
        self._Da = env_spec.action_space.flat_dim
        self._epsilon = epsilon
        self._qf = qf
        self._is_deterministic = False
        super(GreedyPolicy, self).__init__(env_spec)

    # Assumes action spaces are normalized to be the interval [-1, 1]
    @overrides
    def get_action(self, observation):

        """Sample single action based on the observations."""
        return self.get_actions(observation[None])[0], {}

    @overrides
    def get_actions(self, observations):

        #feed_dict = {self._observations_ph: observations}
        probs = np.ones(self._Da , dtype=float) * self._epsilon / self._Da
        q_values = self._qf.eval(observations)
        best_action = np.argmax(q_values)
        probs[best_action] += (1.0 - self._epsilon)

        probs = np.expand_dims(probs, axis=0)
        if self._is_deterministic:
            actions = np.argmax(probs, axis=1)
        else:
            actions = list(map(self.action_space.weighted_sample, probs))

        return actions

    def epsilon_decay(self):
        self._epsilon -= 5e-5
        self._epsilon = max(self._epsilon, 0.1)

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass

