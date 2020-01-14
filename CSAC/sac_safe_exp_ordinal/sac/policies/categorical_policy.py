""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sac.policies import NNPolicy
from sac.misc import tf_utils
from sac.distributions import Categorical

EPS = 1e-6


class CategoricalPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3,
                 name='categorical_policy'
                 ):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.comp_dim
        self._Ba = env_spec.action_space.comp_base
        self._Ds = env_spec.observation_space.flat_dim
        self._is_deterministic = False
        self.name = name
        self.build()

        self._scope_name = (
                tf.get_variable_scope().name + "/" + name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations,
                    name=None, reuse=tf.AUTO_REUSE,
                    ):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = Categorical(
                Dx=self._Da,
                hidden_layers_sizes=self._hidden_layers,
                cond_t_lst=(observations,)
            )
            #p_alls = distribution.p_all
            logp_alls = distribution.logp_all
            actions = [tf.multinomial(logit, 1)for logit in distribution.logits]
            logp_pi = [tf.reduce_sum(tf.one_hot(tf.squeeze(action, axis=1), depth=Da) * logp_all, axis=1)
                       for action, logp_all, Da in zip(actions, logp_alls, self._Da)]

        return [(tf.cast(action,'float32')-b)/b for action, b in zip(actions, self._Ba)], tf.add_n(logp_pi)#,logp_pi,logp_alls

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Categorical(
                Dx=self._Da,
                hidden_layers_sizes=self._hidden_layers,
                cond_t_lst=(self._observations_ph,)
            )

    @overrides
    def get_actions(self, observations):

        feed_dict = {self._observations_ph: observations}

        probs = tf.get_default_session().run(
                self.distribution.p_all, feed_dict)  # 1 x Da

        probs = np.concatenate(probs, axis=1)

        if self._is_deterministic:
            actions = list(map(self.action_space.determined_sample_normalized, probs))
        else:
            actions = list(map(self.action_space.weighted_sample_normalized, probs))


        return actions

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



    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        probs = sess.run(self.distribution.p_all, feeds)

        logger.record_tabular('policy-prob-sum', np.mean([np.mean(np.sum(prob,1)) for prob in probs]))
