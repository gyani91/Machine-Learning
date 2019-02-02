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
