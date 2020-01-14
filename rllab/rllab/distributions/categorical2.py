import theano.tensor as TT
import numpy as np
from .base import Distribution
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

TINY = 1e-8


# def from_onehot_sym(x_var):
#     ret = TT.zeros((x_var.shape[0],), x_var.dtype)
#     nonzero_n, nonzero_a = TT.nonzero(x_var)[:2]
#     ret = TT.set_subtensor(ret[nonzero_n], nonzero_a.astype('uint8'))
#     return ret


def from_onehot(x_var):
    ret = np.zeros((len(x_var),), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical2(Distribution):
    def __init__(self, comp_dim):
        self._comp_dim = comp_dim
        self._slice = [0] + np.cumsum(self._comp_dim).tolist()[:-1]
        self._srng = RandomStreams()

    @property
    def dim(self):
        return self._comp_dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        l_kl_sym = [TT.sum(
            old_prob_var[:, s:s+n] * (TT.log(old_prob_var[:,s:s+n] + TINY) - TT.log(new_prob_var[:,s:s+n] + TINY)),
            axis=-1) for s, n in zip(self._slice, self._comp_dim)]

        return TT.sum(l_kl_sym, axis=0)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]

        l_kl = [np.sum(
            old_prob[:, s:s+n] * (np.log(old_prob[:, s:s+n] + TINY) - np.log(new_prob[:, s:s+n] + TINY)),
            axis=-1) for s, n in zip(self._slice, self._comp_dim)]

        return np.sum(l_kl, axis=0)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars): ## not used for CPO
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        x_var = TT.cast(x_var, 'float32')
        # Assume layout is N * A

        npv = TT.prod([TT.sum(new_prob_var[:,s:s+n] * x_var[:,s:s+n], axis=-1) for s, n in zip(self._slice, self._comp_dim)], axis=0) + TINY
        opv = TT.prod([TT.sum(old_prob_var[:,s:s+n] * x_var[:,s:s+n], axis=-1) for s, n in zip(self._slice, self._comp_dim)], axis=0) + TINY
        return npv / opv

    def entropy(self, info):
        probs = info["prob"]
        l_entropy = [-np.sum(probs[:,s:s+n] * np.log(probs[:,s:s+n] + TINY), axis=1) for s, n in zip(self._slice, self._comp_dim) ]
        return np.sum(l_entropy, axis=0)

    def entropy_sym(self, dist_info_vars):
        prob_var = dist_info_vars["prob"]
        l_entropy_var = [-TT.sum(prob_var[:, s:s + n] * np.log(prob_var[:, s:s + n] + TINY), axis=1) for s, n in
                     zip(self._slice, self._comp_dim)]
        return TT.sum(l_entropy_var, axis=0)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * A
        l_log_likelihood_sym = [TT.log(TT.sum(probs[:,s:s+n] * TT.cast(xc, 'float32'), axis=-1) + TINY)
                            for s, n, xc in zip(self._slice, self._comp_dim, x_var)]
        return TT.sum(l_log_likelihood_sym, axis=0)

    def log_likelihood(self, xs, dist_info):
        probs = dist_info["prob"]
        # Assume layout is N * A
        N = probs.shape[0]
        l_log_likelihood = [np.log(probs[:,s:s+n][np.arange(N), from_onehot(np.asarray(xs[:,s:s+n]))] + TINY)
                            for s,n in zip(self._slice, self._comp_dim)]
        return np.sum(l_log_likelihood, axis=0)

    def sample_sym(self, dist_info):
        probs = dist_info["prob"]
        return self._srng.multinomial(pvals=probs, dtype='uint8')

    @property
    def dist_info_keys(self):
        return ["prob"]
