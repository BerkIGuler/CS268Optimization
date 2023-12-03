import networkx as nx
import matplotlib.pyplot as plt


def barbell_gr():
    plt.figure(figsize=(12,12))
    g = nx.barbell_graph(49, 2)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()


def connected_erdos_random_gr(max_trials=1000):
    found_connected = False
    for _ in range(max_trials):
        g = nx.erdos_renyi_graph(100, 0.03)
        if nx.is_connected(g):
            found_connected = True
            break

    if found_connected is False:
        raise RuntimeError("Could not generate a connected graph")

    plt.figure(figsize=(12, 12))
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    total_nbrs = 0
    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)

    print("average degree:", total_nbrs / 100)


def connected_watts_strogatz_graph():
    """used in https://ceur-ws.org/Vol-3194/paper38.pdf"""
    plt.figure(figsize=(12,12))
    g = nx.connected_watts_strogatz_graph(100, 4, 0.5)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    total_nbrs = 0
    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)

    print(total_nbrs/100)


def rand_regular_gr():
    plt.figure(figsize=(12, 12))
    g = nx.random_regular_graph(3, 100)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    total_nbrs = 0
    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)

    print(total_nbrs / 100)


def barabasi_albert_gr():
    """used in https://ceur-ws.org/Vol-3194/paper38.pdf"""
    plt.figure(figsize=(12, 12))
    g = nx.barabasi_albert_graph(100, 2)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    total_nbrs = 0
    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)

    print(total_nbrs / 100)


def complete_gr():
    plt.figure(figsize=(12, 12))
    g = nx.complete_graph(100)
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()

    total_nbrs = 0
    for n, nbrs in g.adj.items():
        total_nbrs += len(nbrs)

    print(total_nbrs / 100)