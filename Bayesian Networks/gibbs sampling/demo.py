import matplotlib.pyplot as plt
import bprop
import examples_bprop
import sampling


# Load coin flipping Bayes network and create Gibbs sampler.
bn = examples_bprop.bn_naive_bayes()
g = bprop.FactorGraph(bn)
sampler = sampling.GibbsSampler(g)

# Draw 10,000 samples, after a 100-sample burn-in and plot approximate
# cumulative marginals.
marg = sampler.run(10000, burnin=100)
bprop.draw_marginals(marg, markers=False)
plt.show()

# Do the same, when conditioned on having observed heads, tails, heads.
sampler.condition({'X1': 'H', 'X2': 'T', 'X3': 'H'})
marg = sampler.run(10000, burnin=100)
bprop.draw_marginals(marg, markers=False)
plt.show()
