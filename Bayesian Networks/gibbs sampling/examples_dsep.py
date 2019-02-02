import core


# +---+     +---+
# | X |     | Y |
# +---+     +---+
def bn_independent():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y'])
    return g


# +---+     +---+
# | X |---->| Y |
# +---+     +---+
def bn_dependent():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y'])
    g.add_edge('X', 'Y')
    return g


# +---+     +---+     +---+
# | X |---->| Y |---->| Z |
# +---+     +---+     +---+
def bn_chain():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Y'), ('Y', 'Z')])
    return g


# +---+     +---+     +---+
# | Y |<----| X |---->| Z |
# +---+     +---+     +---+
def bn_naive_bayes():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Y'), ('X', 'Z')])
    return g


# +---+     +---+     +---+
# | X |---->| Z |<----| Y |
# +---+     +---+     +---+
def bn_v_structure():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Z'), ('Y', 'Z')])
    return g


# +---+
# | X |
# +---+
#   |
#   v
# +---+     +---+
# | Y |<----| W |
# +---+     +---+
#   |         |
#   v         |
# +---+       |
# | Z |<------+
# +---+
def bn_koller():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'W', 'Z'])
    g.add_edges_from([('X', 'Y'), ('W', 'Y'), ('W', 'Z'), ('Y', 'Z')])
    return g


# +------------+         +---------+
# | Earthquake |         | Burglar |
# +------------+         +---------+
#     |   |                   |
#     |   |     +-------+     |
#     |   +---->| Alarm |<----+
#     v         +-------+
# +-------+         |
# | Radio |         |
# +-------+         v
#               +-------+
#               | Phone |
#               +-------+
def bn_earthquake():
    g = core.BayesNet()
    g.add_nodes_from(['Earthquake', 'Burglar', 'Alarm', 'Radio', 'Phone'])
    g.add_edges_from([('Earthquake', 'Radio'),
                      ('Earthquake', 'Alarm'),
                      ('Burglar', 'Alarm'),
                      ('Alarm', 'Phone')])
    return g
