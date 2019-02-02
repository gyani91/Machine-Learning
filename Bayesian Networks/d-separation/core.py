"""
PAI: Homework 2 --- d-separation in Bayesian networks
=====================================================
"""

import networkx as nx


EDGE_COLOR = '#bbbbbb'
EDGE_WIDTH = 2
NODE_SIZE = 3000
NODE_BORDER_COLOR = EDGE_COLOR
NODE_BORDER_WIDTH = 3
NODE_COLOR_NORMAL = '#3492d9'
NODE_COLOR_SOURCE = '#2cb64e'
NODE_COLOR_OBSERVED = '#d96b34'
NODE_COLOR_REACHABLE = NODE_COLOR_SOURCE
NODE_SHAPE_SOURCE = 'd'
LABEL_COLOR = '#111111'


class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain


class BayesNet(nx.DiGraph):
    """Represents a Bayesian network as a directed graph."""

    def __init__(self):
        super(BayesNet, self).__init__()

    def get_ancestors(self, variables):
        """Get all ancestors of the given variables.

        Arguments
        ---------
        variables : iterable of str

        Returns
        -------
        A set with the ancestors.
        """
        to_visit = set(variables)
        ancestors = set()
        while to_visit:
            variable = to_visit.pop()
            if variable not in ancestors:
                ancestors.add(variable)
                to_visit |= set(self.predecessors(variable))
        return ancestors

    def get_reachable(self, x, observed=None, plot=False):
        """Get all nodes that are reachable from x, given the observed nodes.

        Arguments
        ---------
        x : str
            Source node.

        observed : iterable of str
            A set of observed variables. Defaults to None (no observations)

        plot : bool
            If True, plot network with distinguishing colors for observable,
            reachable, and d-separated nodes.

        Returns
        -------
        The set of reachable nodes.
        """
        if observed is None:
            observed = []
        observed = set(observed)
        assert x in self.nodes()
        assert observed <= set(self.nodes())
        # First, find all ancestors of observed set.
        ancestors = self.get_ancestors(observed)
        # Then, perform a search for reachable variables starting from x.
        # Nodes to be visited are stored as tuples with the following elements:
        #         * the variable
        #         * True, if the variable was reached via an incoming edge,
        #           False, if it was reached via an outgoing edge.
        # Any variable that is reached through an active path is stored in
        # reachable.
        to_visit = set([(x, False)])
        visited = set()
        reachable = set()
        while to_visit != set():
            current = to_visit.pop()
            variable, trail_entering = current
            if current in visited:
                continue
            if variable not in observed:
                reachable.add(variable)
            visited.add(current)
            # <--- V
            if not trail_entering and variable not in observed:
                # <--- V <---
                for predecessor in self.predecessors(variable):
                    to_visit.add((predecessor, False))
                # <--- V --->
                for successor in self.successors(variable):
                    to_visit.add((successor, True))
            # ---> V
            elif trail_entering:
                # ---> V --->
                if variable not in observed:  # only successors blocked.
                    for successor in self.successors(variable):
                        to_visit.add((successor, True))
                # ---> V <---
                elif variable in ancestors:
                    for predecessor in self.predecessors(variable):
                        to_visit.add((predecessor, False))
        # Just a convention to not return the query node.
        reachable.discard(x)
        # Optionally plot.
        if plot:
            self.draw(x, observed, reachable)
        return reachable

    def draw(self, x=None, observed=None, dependent=None):
        """Draw the Bayesian network.

        Arguments
        ---------
        x : str
            The source variable.

        observed : iterable of str
            The variables on which we condition.

        dependent : iterable of str
            The variables which are dependent of ``x`` given ``observed``.
        """
        pos = nx.spectral_layout(self)
        nx.draw_networkx_edges(self, pos,
                               edge_color=EDGE_COLOR,
                               width=EDGE_WIDTH)
        rest = list(
            set(self.nodes()) - set([x]) - set(observed) - set(dependent))
        if rest:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=rest,
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_NORMAL)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if x:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=[x],
                                         node_size=3000,
                                         node_color=NODE_COLOR_SOURCE,
                                         node_shape=NODE_SHAPE_SOURCE)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if observed:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(observed),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_OBSERVED)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if dependent:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(dependent),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_REACHABLE)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        nx.draw_networkx_labels(self, pos, font_color=LABEL_COLOR)
