import networkx as nx
import numpy as np
import logging


logger = logging.getLogger()


def _get_avg_nbrs(g):
    total_nbrs = 0

    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)
    avg_nbrs = total_nbrs / len(g)

    return avg_nbrs


def connected_erdos_random_gr(num_nodes, degree, max_trials=100000):
    found_connected = False
    increment = 0.00001
    p = 0.01
    for _ in range(max_trials):
        g = nx.erdos_renyi_graph(num_nodes, p)
        if nx.is_connected(g) and abs(_get_avg_nbrs(g) - degree) < 0.01:
            found_connected = True
            break
        p += increment

    if found_connected is False:
        raise RuntimeError("Could not generate a connected graph with desired average degree")

    return g


def get_weight_matrix(num_nodes, topology="fc", degree=None):
    if topology == "fc":
        weight_matrix = np.array([1 / num_nodes] * num_nodes ** 2).reshape(num_nodes, num_nodes)
    elif topology == "regular" and degree is not None:
        g = nx.random_regular_graph(d=degree, n=num_nodes)
        identity = np.identity(num_nodes)
        weight_matrix = nx.adjacency_matrix(g).todense() + identity
        weight_matrix = weight_matrix / (degree + 1)
    elif topology == "small_world" and degree is not None:
        if degree % 2 == 1:
            logger.warning("effective degree cannot be odd")
        g = nx.connected_watts_strogatz_graph(num_nodes, degree, 0.5)
        weight_matrix = np.identity(num_nodes) + nx.adjacency_matrix(g).todense()
        weight_matrix /= np.sum(weight_matrix, axis=0)
    elif topology == "erdos" and degree is not None:
        g = connected_erdos_random_gr(num_nodes, degree)
        wm = nx.adjacency_matrix(g).todense() + np.identity(num_nodes)
        weight_matrix = wm / np.sum(wm, axis=0)
    elif topology == "barabasi" and degree is not None:
        logger.warning("graph degree is approximate for barabasi albert")
        g = nx.barabasi_albert_graph(num_nodes, int(degree / 2))
        logger.warning(f"Desired Degree: {degree}, actual: {_get_avg_nbrs(g)}")
        wm = nx.adjacency_matrix(g).todense() + np.identity(num_nodes)
        weight_matrix = wm / np.sum(wm, axis=0)
    else:
        raise NotImplemented

    # assert np.sum(np.sum(weight_matrix, axis=0) - np.ones(num_nodes)) < 0.00000001, "not doubly stochastic"
    # assert np.sum(np.sum(weight_matrix, axis=0) - np.ones(num_nodes)) < 0.00000001, "not doubly stochastic"

    return weight_matrix
