""" Multivariate normal distribution with mean and std deviation outputted by a neural net """

import tensorflow as tf
import numpy as np

from sac.misc.mlp import mlp

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class Categorical(object):
    def __init__(
            self,
            Dx,
            hidden_layers_sizes=(100, 100),
            cond_t_lst=(),
    ):
        self._cond_t_lst = cond_t_lst
        self._layer_sizes = list(hidden_layers_sizes) + [Dx]

        self._Dx = Dx

        self._create_placeholders()
        self._create_graph()

    def _create_placeholders(self):
        self._N_pl = tf.placeholder(
            tf.int32,
            shape=(),
            name='N',
        )

    def _create_graph(self):

        logits = mlp(
            inputs=self._cond_t_lst,
            layer_sizes=self._layer_sizes,
            output_nonlinearity=None,
        )
        self._logits = logits
        self._probs = tf.nn.softmax(logits)
        self._logprobs = tf.nn.log_softmax(logits)

    @property
    def logits(self):
        return self._logits

    @property
    def p_all(self):
        return self._probs

    @property
    def logp_all(self):
        return self._logprobs


