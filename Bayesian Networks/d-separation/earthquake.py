import matplotlib.pylab as plt
import core


g = core.BayesNet()
g.add_nodes_from(['Earthquake', 'Burglar', 'Alarm', 'Radio', 'Phone'])
g.add_edges_from([('Earthquake', 'Radio'),
                  ('Earthquake', 'Alarm'),
                  ('Burglar', 'Alarm'),
                  ('Alarm', 'Phone')])
g.get_reachable('Radio', plot=True)
plt.show()
g.get_reachable('Radio', ['Phone'], plot=True)
plt.show()
g.get_reachable('Radio', ['Phone', 'Earthquake'], plot=True)
plt.show()
