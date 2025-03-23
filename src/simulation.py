import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .models import Investor
from .utils import setup_logging
from .network import update_imitation


def initialize_investors(num_investors):
    investors = []
    for i in range(num_investors):
        initial_capital = np.random.uniform(100, 1000)  # Random capital between 100 and 1000.
        investor = Investor(i, initial_capital)
        investors.append(investor)
        logging.info(f"Initialized Investor {i} with capital {initial_capital:.2f}")
    return investors


def update_investors(investors, dt=1, mu=0, sigma=1):
    for inv in investors:
        prev_perf = inv.performance
        inv.update_performance(dt, mu, sigma)
        logging.info(f"Investor {inv.id} updated performance from {prev_perf:.2f} to {inv.performance:.2f}")


def run_simulation(num_investors=10, time_steps=50, alpha=0.01, beta=0.001):
    setup_logging()
    investors = initialize_investors(num_investors)

    # Create a directed graph to capture imitation relationships.
    G = nx.DiGraph()
    for inv in investors:
        G.add_node(inv.id, performance=inv.performance)

    for t in range(time_steps):
        logging.info(f"Starting time step {t+1}")
        update_investors(investors)
        # Update imitation events and network structure.
        G = update_imitation(investors, G, alpha, beta)
        for inv in investors:
            G.nodes[inv.id]['performance'] = inv.performance
        logging.info(f"Completed time step {t+1}")

    # Display the final investor imitation network.
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Investor Imitation Network")
    plt.show()


if __name__ == '__main__':
    run_simulation()
