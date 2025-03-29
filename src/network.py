import numpy as np
import logging


def validate_investors(investors):
    """
    Validates that each investor in the list has the required attributes:
    'performance', 'capital', 'history', and 'id'.
    Raises ValueError if any investor is missing any attribute.
    """
    for investor in investors:
        for attr in ['performance', 'capital', 'history', 'id']:
            if not hasattr(investor, attr):
                raise ValueError(f"Investor {investor} is missing required attribute '{attr}'")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_imitation(investors, G, alpha=0.01, beta=0.001):
    """
    For each pair of investors, compute the imitation probability:
    P(i copies j) = sigmoid(alpha * (j.performance - i.performance) + beta * (j.capital - i.capital)).
    If a random draw is below this probability, investor i imitates j.

    Imitation is modeled by blending investor i's performance toward investor j's performance,
    and an edge is added (or updated) in the network graph with weight equal to the imitation probability.

    Parameters:
        investors (list): List of investor objects. Each investor should have the following attributes:
            - performance (float): Current performance of the investor.
            - capital (float): Capital of the investor.
            - history (list): A list storing the performance history (with the last element representing the current performance).
            - id (any): A unique identifier for the investor.
        G (networkx.DiGraph): A directed network graph (from networkx) where nodes represent investors.
        alpha (float): Scaling factor for performance difference (default: 0.01).
        beta (float): Scaling factor for capital difference (default: 0.001).

    Returns:
        G (networkx.DiGraph): Updated network graph with imitation edges.
    """
    # Validate that each investor has the required attributes
    validate_investors(investors)

    n = len(investors)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Use descriptive variable names
            investor_i = investors[i]
            investor_j = investors[j]
            try:
                # Compute imitation probability using the sigmoid function
                # The formula factors in the differences in performance and capital
                prob = sigmoid(alpha * (investor_j.performance - investor_i.performance) + beta * (investor_j.capital - investor_i.capital))
                # If a random draw is below the imitation probability, perform imitation
                if np.random.rand() < prob:
                    weight = prob  # Use the probability as the blending weight
                    old_perf = investor_i.performance
                    investor_i.performance = (1 - weight) * investor_i.performance + weight * investor_j.performance

                    # Ensure history is not empty before updating
                    if investor_i.history:
                        investor_i.history[-1] = investor_i.performance
                    else:
                        raise ValueError(f"Investor {investor_i.id} history is empty. Cannot update performance history.")

                    # Update network: add/update a directed edge from investor_i to investor_j with the imitation weight
                    if G.has_edge(investor_i.id, investor_j.id):
                        G[investor_i.id][investor_j.id]['weight'] = weight
                    else:
                        G.add_edge(investor_i.id, investor_j.id, weight=weight)

                    logging.info(f"Investor {investor_i.id} imitated Investor {investor_j.id}: prob={prob:.4f}, performance {old_perf:.2f} -> {investor_i.performance:.2f}")
            except Exception as e:
                logging.error(f"Error processing imitation from investor {investor_i.id} to {investor_j.id}: {str(e)}")
                continue
    return G
