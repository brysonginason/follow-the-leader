import numpy as np
import logging


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_imitation(investors, G, alpha=0.01, beta=0.001):
    """
    For each pair of investors, compute the imitation probability:
    P(i copies j) = sigmoid( alpha*(j.performance - i.performance) + beta*(j.capital - i.capital) ).
    If a random draw is below this probability, investor i imitates j.

    Imitation is modeled by blending i's performance toward j's performance,
    and an edge is added (or updated) in the network graph with weight equal to the imitation probability.
    """
    n = len(investors)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inv_i = investors[i]
            inv_j = investors[j]
            # Compute imitation probability
            prob = sigmoid(alpha * (inv_j.performance - inv_i.performance) + beta * (inv_j.capital - inv_i.capital))
            if np.random.rand() < prob:
                # Imitation event: update investor i's performance by copying j partially.
                weight = prob  # Use the probability as the weight.
                old_perf = inv_i.performance
                inv_i.performance = (1 - weight) * inv_i.performance + weight * inv_j.performance
                # Update the latest performance in history.
                inv_i.history[-1] = inv_i.performance
                # Update network: add/update a directed edge from i to j with the imitation weight.
                if G.has_edge(inv_i.id, inv_j.id):
                    G[inv_i.id][inv_j.id]['weight'] = weight
                else:
                    G.add_edge(inv_i.id, inv_j.id, weight=weight)
                logging.info(f"Investor {inv_i.id} imitated Investor {inv_j.id}: prob={prob:.4f}, performance {old_perf:.2f} -> {inv_i.performance:.2f}")
    return G
