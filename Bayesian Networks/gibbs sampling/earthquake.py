import matplotlib.pylab as plt
import core
import bprop
import sampling


# Create earthquake Bayesian network.
bn = core.BayesNet()
bn.add_variable('Earthquake', (0, 1))
bn.add_variable('Burglar', (0, 1))
bn.add_variable('Radio', (0, 1))
bn.add_variable('Alarm', (0, 1))
bn.add_variable('Phone', (0, 1))
bn.add_cpt(None, 'Earthquake',
           {0: 0.999,
            1: 0.001})
bn.add_cpt(None, 'Burglar',
           {0: 0.999,
            1: 0.001})
bn.add_cpt(('Burglar', 'Earthquake'), 'Alarm',
           {(0, 0, 0): 0.999,
            (0, 0, 1): 0.001,
            (1, 0, 0): 0.00999,
            (1, 0, 1): 0.99001,
            (0, 1, 0): 0.98901,
            (0, 1, 1): 0.01099,
            (1, 1, 0): 0.0098901,
            (1, 1, 1): 0.9901099})
bn.add_cpt('Alarm', 'Phone',
           {(0, 1): 0,
            (0, 0): 1,
            (1, 0): 0.3,
            (1, 1): 0.7})
bn.add_cpt('Earthquake', 'Radio',
           {(0, 1): 0,
            (0, 0): 1,
            (1, 0): 0.5,
            (1, 1): 0.5})

# Create factor graph and Gibbs sampler.
g = bprop.FactorGraph(bn)
sampler = sampling.GibbsSampler(g)

# Draw 20,000 samples and only keep every 50th sample after a burn-in period of
# 100 samples.
marg = sampler.run(20000, burnin=100, step=50)
bprop.draw_marginals(marg, markers=False)
plt.show()

# Condition on Phone = 1 and run again.
sampler.condition({'Phone': 1})
marg = sampler.run(20000, burnin=100, step=50)
bprop.draw_marginals(marg, markers=False)
plt.show()

# Additionally, condition on Radio = 1 and run again.
g.condition({'Radio': 1})
marg = sampler.run(20000, burnin=100, step=50)
bprop.draw_marginals(marg, markers=False)
plt.show()
