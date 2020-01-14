""" Multivariate normal distribution with mean and std deviation outputted by a neural net """

import tensorflow as tf
import numpy as np

from sac.misc.mlp import mlp

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20
EPS = 1e-6

class Categorical(object):
    def __init__(
            self,
            Dx,
            hidden_layers_sizes=(100, 100),
            cond_t_lst=(),
    ):
        self._cond_t_lst = cond_t_lst
        self._layer_sizes = list(hidden_layers_sizes) + [np.sum(Dx)]

        self._Dx = Dx
        self._actnum = len(self._Dx)
        self._slice = [0] + np.cumsum(self._Dx).tolist()

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
            output_nonlinearity=tf.nn.sigmoid,
        )
        l_logits = [logits[:,self._slice[i]:self._slice[i+1]] for i in range(self._actnum)]
        self._logits = []
        for i in range(self._actnum):
            up_tri = tf.constant(np.triu(np.ones((self._Dx[i], self._Dx[i]), dtype=np.float32)))
            low_tri = tf.constant(np.tril(np.ones((self._Dx[i], self._Dx[i]), dtype=np.float32), -1))
            ordinal_logits = tf.matmul(tf.log(l_logits[i]+EPS), up_tri) + tf.matmul(tf.log(1-l_logits[i]+EPS), low_tri)
            self._logits.append(ordinal_logits)
        #self._logits = [logits[:,self._slice[i]:self._slice[i+1]] for i in range(self._actnum)]
        self._probs = [tf.nn.softmax(logit) for logit in self._logits]
        self._logprobs = [tf.nn.log_softmax(logit) for logit in self._logits]
        #self._states = [tf.nn.sigmoid(logits[self._slice[i]:self._slice[i+1]]) for i in range(self._actnum)]

        #self._logprobs = [tf.nn.softmax(logits[self._slice[i]:self._slice[i+1]]) for i in range(self._actnum)]

    @property
    def logits(self):
        return self._logits

    @property
    def p_all(self):
        return self._probs

    @property
    def logp_all(self):
        return self._logprobs


