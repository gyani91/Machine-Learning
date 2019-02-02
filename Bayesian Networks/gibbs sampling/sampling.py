import numpy as np
import numpy.random as npr
import bprop


def cumulative_average(array, step=1):
    """Compute cumulative average of ``array``.

    The result is an array cumulative_avg, s.t., for all 0 <= i < len(array)

        cumulative_avg[i] = (array[0] + array[1] + ... + array[i]) / (i + 1).

    Arguments
    ---------
    array : iterable
        Input array.

    Returns
    -------
    A (k + 1) x 1 array of cumulative averages.
    """
    avg = np.cumsum(np.asarray(array), dtype=float)
    avg /= np.arange(1, len(avg) + 1)
    return avg


class GibbsSampler:
    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.update_fgraph()

    def update_fgraph(self):
        """Should be called when the associated factor graph is updated."""
        self.vs = self.fgraph.vs
        self.vobs = self.fgraph.vobs

    def condition(self, observations):
        """Convenience method. Same as ``bprob.FactorGraph.condition``."""
        self.fgraph.condition(observations)
        self.update_fgraph()

    def sample_var(self, v, state):
        """Sample a value of variable ``v`` from its posterior given ``state``.

        We need to only consider the values of variables in ``state`` that
        belong to the Markov blanket of ``v``, equivalently, all variables that
        participate in factors that are neighbors of ``v`` in the factor graph.

        Arguments
        ---------
        v : str
            Name of the variable to be updated.

        state : dict
            Current state of the Gibbs sampler.

        Returns
        -------
        A randomly sampled value of ``v`` from the posterior P(v | state\{v}).
        """
        v_domain = self.vs[v].domain
        prob = np.zeros(len(v_domain))
        for d in v_domain:
            for fnode in self.vs[v].neighbors:
                comb = []
                for fnode_var in fnode.variables:
                    if fnode_var == v:
                        comb.append(d)
                    else:
                        comb.append(state[fnode_var])
                prob[d] += fnode.table[tuple(comb)]
        prob = bprop.normalize(prob)
        return npr.choice(v_domain, p=np.exp(prob))

    def run(self, niter, burnin=0, step=1, init_state=None):
        """Run a Gibbs sampler to estimate marginals using ``niter`` samples.

        Optionally, use a burn-in period during which samples are discarded,
        and specify (part of) the starting state.

        Arguments
        ---------
        niter : int
            Number of samples to be returned.

        burnin : int
            Length of burn-in period.

        step : int
            Every ``step``-th point will be considered in the average.

            For example, if ``step=5``, then the following samples will be
            averaged: 0, 5, 10, 15, ...

        init_state : dict
            Starting state. Can be specified partially by only providing
            initial values for a subset of all variables.

        Returns
        -------
        A tuple of computed marginals, variable domains, and observations,
        same as that returned by ``bprob.FactorGraph.run_bp``.
        """
        assert burnin < niter
        variables = self.vs.keys()
        samples = {v: [] for v in variables}
        # If not specified, the initial value of each variable is drawn
        # uniformly at random.
        state = {v: npr.choice(vnode.domain) for v, vnode in self.vs.items()}
        if init_state is not None:
            state.update(init_state)
        n_iterations = niter + burnin
        for it in range(n_iterations):
            variable = npr.choice(variables)
            state[variable] = self.sample_var(variable, state)
            # Ignore burnin samples, otherwise take every ``step``-th sample.
            if it >= burnin and (it - burnin) % step == 0:
                for v in variables:
                    samples[v].append(state[v])
        marginals = self.get_marginals(samples)
        domains = {v.name: v.orig_domain for v in self.vs.values()}
        return (marginals, domains, self.fgraph.vobs)

    def get_marginals(self, samples):
        """Compute approximate marginals.

        For every value a variable can take, a binary indicator array is
        created that indicates at which iterations the variable had that value.

        Arguments
        ---------
        samples : dict
            Dictionary that maps each variable to its samples as produced by
            the Gibbs sampler.

        Returns
        -------
        A dictionary that maps each variable v to a N x |domain(v)| array,
        where the i-th row holds the estimated marginals after i samples.
        """
        niter = len(samples.values()[0])
        assert niter >= 1
        marginals = {
            v: np.zeros((niter, len(self.vs[v].domain))) for v in samples}
        for v in samples:
            v_samples = np.array(samples[v])
            for i, d in enumerate(self.vs[v].domain):
                bin_array = np.zeros((niter, 1))
                bin_array[v_samples == d] = 1
                marginals[v][:, i] = cumulative_average(bin_array)
        return marginals
