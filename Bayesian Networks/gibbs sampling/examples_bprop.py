import core


def bn_vstruct():
    g = core.BayesNet()
    g.add_variable('X', (0, 1))
    g.add_variable('Y', (0, 1))
    g.add_variable('Z', (0, 1))
    g.add_cpt(None, 'X',
              {0: 0.001,
               1: 0.999})
    g.add_cpt(None, 'Y',
              {0: 0.001,
               1: 0.999})
    g.add_cpt(('X', 'Y'), 'Z',
              {(0, 0, 0): 0.99,
               (0, 0, 1): 0.01,
               (0, 1, 0): 0.99,
               (0, 1, 1): 0.01,
               (1, 0, 0): 0.99,
               (1, 0, 1): 0.01,
               (1, 1, 0): 0.001,
               (1, 1, 1): 0.999})
    return g


# The Naive Bayes network from Problem Set 2, Exercise 2.
def bn_naive_bayes():
    g = core.BayesNet()
    g.add_variable('Coin', ('a', 'b', 'c'))
    g.add_variable('X1', ('H', 'T'))
    g.add_variable('X2', ('H', 'T'))
    g.add_variable('X3', ('H', 'T'))
    g.add_cpt(None, 'Coin', {'a': 1.0 / 3, 'b': 1.0 / 3, 'c': 1.0 / 3})
    g.add_cpt('Coin', 'X1',
              {('a', 'H'): 0.2,
               ('a', 'T'): 0.8,
               ('b', 'H'): 0.6,
               ('b', 'T'): 0.4,
               ('c', 'H'): 0.8,
               ('c', 'T'): 0.2})
    g.add_cpt('Coin', 'X2',
              {('a', 'H'): 0.2,
               ('a', 'T'): 0.8,
               ('b', 'H'): 0.6,
               ('b', 'T'): 0.4,
               ('c', 'H'): 0.8,
               ('c', 'T'): 0.2})
    g.add_cpt('Coin', 'X3',
              {('a', 'H'): 0.2,
               ('a', 'T'): 0.8,
               ('b', 'H'): 0.6,
               ('b', 'T'): 0.4,
               ('c', 'H'): 0.8,
               ('c', 'T'): 0.2})
    return g


# The earthquake network from Problem Set 4.
def bn_earthquake():
    g = core.BayesNet()
    g.add_variable('Earthquake', (0, 1))
    g.add_variable('Burglar', (0, 1))
    g.add_variable('Radio', (0, 1))
    g.add_variable('Alarm', (0, 1))
    g.add_variable('Phone', (0, 1))
    g.add_cpt(None, 'Earthquake',
              {0: 0.999,
               1: 0.001})
    g.add_cpt(None, 'Burglar',
              {0: 0.999,
               1: 0.001})
    g.add_cpt(('Burglar', 'Earthquake'), 'Alarm',
              {(0, 0, 0): 0.999,
               (0, 0, 1): 0.001,
               (1, 0, 0): 0.00999,
               (1, 0, 1): 0.99001,
               (0, 1, 0): 0.98901,
               (0, 1, 1): 0.01099,
               (1, 1, 0): 0.0098901,
               (1, 1, 1): 0.9901099})
    g.add_cpt('Alarm', 'Phone',
              {(0, 1): 0,
               (0, 0): 1,
               (1, 0): 0.3,
               (1, 1): 0.7})
    g.add_cpt('Earthquake', 'Radio',
              {(0, 1): 0,
               (0, 0): 1,
               (1, 0): 0.5,
               (1, 1): 0.5})
    return g
