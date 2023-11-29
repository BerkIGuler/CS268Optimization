import networkx as nx
from random import sample
from math import sqrt
import matplotlib.pyplot as plt

def calculate_cost(weights, edge_cost_function, node_cost_function):
    n = len(weights)

    edge_cost = sum([edge_cost_function(x, i) for i, x in enumerate(weights)])
    node_cost = node_cost_function((n * (n - 1)) // 2)
    return edge_cost + node_cost

def get_adjacency_coordinates(n, num_vals):
    row, skip, = 0, 1
    for coordinate in range(num_vals):
        if (coordinate + skip) // n != row:
            row += 1
            skip += row + 1
        yield (coordinate + skip) // n, (coordinate + skip) % n

def generate_graph(n, weights, nodes):
    G = nx.empty_graph()
    for i in nodes:
        G.add_node(i, network = nodes[i])

    for i, (u, v) in enumerate(get_adjacency_coordinates(n, len(weights))):
        G.edges[u, v]['weight'] = weights[i]

    return G

def get_time_and_loss_of_function(n, weights):
    G = generate_graph(n, weights, None)
