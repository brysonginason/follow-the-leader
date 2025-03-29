import argparse
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from models import Investor
from utils import setup_logging
from network import update_imitation
from typing import List, Tuple

# Try importing tqdm for a progress bar; if not available, proceed without it.
try:
    from tqdm import tqdm
    progress_bar_available = True
except ImportError:
    progress_bar_available = False


def initialize_investors(num_investors: int) -> List[Investor]:
    """
    Initialize a list of investors with random capital between 100 and 1000.

    Parameters:
        num_investors (int): Number of investors to initialize.

    Returns:
        List[Investor]: A list of Investor objects.
    """
    investors: List[Investor] = []
    for i in range(num_investors):
        initial_capital = np.random.uniform(100, 1000)  # Random capital between 100 and 1000.
        investor = Investor(i, initial_capital)
        investors.append(investor)
        logging.info(f"Initialized Investor {i} with capital {initial_capital:.2f}")
    return investors


def update_investors(investors: List[Investor], dt: float = 1, mu: float = 0, sigma: float = 1) -> None:
    """
    Update each investor's performance using a simple Brownian motion model.

    Parameters:
        investors (List[Investor]): List of investor objects.
        dt (float): Time step for the update.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
    """
    for inv in investors:
        try:
            prev_perf = inv.performance
            inv.update_performance(dt, mu, sigma)
            logging.info(f"Investor {inv.id} updated performance from {prev_perf:.2f} to {inv.performance:.2f}")
        except Exception as e:
            logging.exception(f"Error updating investor {inv.id}: {str(e)}")


def run_simulation(num_investors: int = 10, time_steps: int = 50, alpha: float = 0.01, beta: float = 0.001) -> Tuple[nx.DiGraph, List[Investor]]:
    """
    Run the simulation of investor performance and imitation over a number of time steps.

    Parameters:
        num_investors (int): Number of investors to simulate.
        time_steps (int): Number of time steps in the simulation.
        alpha (float): Scaling factor for performance difference in imitation.
        beta (float): Scaling factor for capital difference in imitation.

    Returns:
        Tuple[nx.DiGraph, List[Investor]]: The final imitation network graph and list of investors.
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    setup_logging()
    logging.info("Starting simulation...")
    investors = initialize_investors(num_investors)

    # Create a directed graph to capture imitation relationships.
    G = nx.DiGraph()
    for inv in investors:
        G.add_node(inv.id, performance=inv.performance)

    iterator = range(time_steps)
    if progress_bar_available:
        iterator = tqdm(iterator, desc="Simulating time steps")

    for t in iterator:
        logging.info(f"Starting time step {t+1}")
        try:
            update_investors(investors)
            # Update imitation events and network structure.
            G = update_imitation(investors, G, alpha, beta)
            for inv in investors:
                G.nodes[inv.id]['performance'] = inv.performance
            logging.info(f"Completed time step {t+1}")
        except Exception as e:
            logging.exception(f"Error during simulation at time step {t+1}: {str(e)}")

    logging.info("Simulation completed.")
    return G, investors


def visualize_network(G: nx.DiGraph) -> None:
    """
    Visualize the final investor imitation network.

    Parameters:
        G (nx.DiGraph): The network graph representing imitation relationships.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Investor Imitation Network")
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the investor simulation.")
    parser.add_argument("--num_investors", type=int, default=10, help="Number of investors to simulate.")
    parser.add_argument("--time_steps", type=int, default=50, help="Number of time steps to simulate.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha scaling factor for imitation.")
    parser.add_argument("--beta", type=float, default=0.001, help="Beta scaling factor for imitation.")
    args = parser.parse_args()

    G, investors = run_simulation(args.num_investors, args.time_steps, args.alpha, args.beta)
    visualize_network(G)
