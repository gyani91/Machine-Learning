import matplotlib.pyplot as plt
import examples_dsep

g = examples_dsep.bn_v_structure()
g.get_reachable('X', plot=True)
plt.show()
g.get_reachable('X', ['Z'], plot=True)
plt.show()
