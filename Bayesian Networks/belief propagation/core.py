from collections import defaultdict
import networkx as nx
from conf import *


EPS = 1e-10


def is_valid_cpt(table):
    """Check that ``table`` contains valid conditional prob. distributions.

    Note that the conditional distributions are defined over the last variable
    in the tuples, while all other variables are conditioned on.
    """
    probabilities = defaultdict(float)
    for combination, value in table.items():
        if value < 0 or value > 1:
            return False
        probabilities[combination[:-1]] += value
    return all(abs(total - 1) <= EPS for total in probabilities.values())


class Variable:
    """A Bayesian network variable."""
    def __init__(self, name, domain, parents=None, cpt=None):
        self.name = name
        self.domain = domain
        self.parents = parents
        self.cpt = cpt


class BayesNet(nx.DiGraph):
    """A Bayesian network as a directed graph."""

    def __init__(self):
        super(BayesNet, self).__init__()
        self.vs = {}  # Variables of the network indexed by name.

    def add_variable(self, name, domain):
        """Add a variable node with the given name to the network.

        Arguments
        ---------
        name : str
            Variable name.

        domain : iterable
            Values the variable can take.
        """
        name = str(name)
        if name in self.vs:
            raise RuntimeError("Variable '{0}' already defined".format(name))
        v = Variable(name, domain, None, None)
        self.vs[name] = v

    def add_cpt(self, parents, variable, table):
        """Add a conditional probability table (CPT) to the network.

        Arguments
        ---------
        parents : iterable of str
            Parents of ``variable`` in the network.

        variable : str
            Variable for which the CPT is given.

        table : dict
            The CPT as a dictionary from tuples of variable values to
            conditional probabilities in the following form:

              { (vp_1, vp_2, ... , v_v): p, ...}

            In the above, p is the conditional probability of v having value
            v_v, given that its parents have values vp_1, vp_2, etc.
        """
        if parents is None:
            parents = ()
        elif isinstance(parents, str):
            parents = (parents,)
        else:
            parents = tuple(parents)
        for v in list(parents) + [variable]:
            if v not in self.vs:
                raise RuntimeError("Unknown variable '{0}'".format(v))
        # For CPTs with no parents, accept non-iterables as table keys for user
        # convenience, but convert them to single-element tuples here.
        newtable = {}
        for c, v in table.items():
            try:
                newtable[tuple(c)] = v
            except TypeError:
                newtable[(c,)] = v
        table = defaultdict(lambda: 0.5, newtable)
        if not is_valid_cpt(table):
            raise RuntimeError('Invalid CPT')
        self.vs[variable].parents = parents
        self.vs[variable].cpt = table
        for parent in parents:
            if not self.has_edge(parent, variable):
                self.add_edge(parent, variable)

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
            The variables which are dependent on ``x`` given ``observed``.
        """
        pos = nx.spectral_layout(self)
        nx.draw_networkx_edges(self, pos,
                               edge_color=EDGE_COLOR,
                               width=EDGE_WIDTH)
        if x or observed or dependent:
            rest = list(
                set(self.nodes()) - set([x]) - set(observed) - set(dependent))
        else:
            rest = self.nodes()
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
