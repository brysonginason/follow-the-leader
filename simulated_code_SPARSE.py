import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)

# Generates d independent Brownian motions of size n with drift, volatility
def brownian_motion(steps, n, mu, sigma, dt=1):
    """
    Parameters:
        steps (int): Number of time steps.
        n (int): Number of independent Brownian motions.
        mu (float): Drift term (default is 0).
        sigma (float): Volatility term (default is 1).
        dt (float): Time step size (default is 1).
    """
    mu = np.asarray(mu).reshape(n, 1)      # Ensure mu is (n,1) for broadcasting
    sigma = np.asarray(sigma).reshape(n, 1) # Ensure sigma is (n,1) for broadcasting

    dW = np.sqrt(dt) * np.random.randn(n, steps)  # Standard Brownian increments
    W = np.cumsum(sigma * dW, axis=1)  # Scale by volatility and integrate
    t = np.arange(1, steps + 1) * dt  # Time steps
    drift = mu * t  # Linear drift component

    return W + drift  # Add drift

def replace_diagonal_with_minus_one(matrix):
        np.fill_diagonal(matrix, -1)
        return matrix
# Update wealth based on Brownian motion.
def update_wealth(wealth, bm_vector):
    """
    Parameters:
        wealth (numpy.ndarray): Vector of wealth for each player.
        bm_vector (numpy.ndarray): Brownian motion vector (n x steps).
    """
    return wealth * (1 + bm_vector[:, -1])

def topk_with_forced_index(prob_vector, i):
    n = len(prob_vector)
    k = max(5, int(np.ceil(n / 100)))

    # Get indices of top-k values
    topk_indices = np.argpartition(prob_vector, -k)[-k:]

    # Make sure index i is included
    # if i not in topk_indices:
        # topk_indices = np.append(topk_indices, i)

    # Create filtered vector
    filtered = np.zeros_like(prob_vector)
    filtered[topk_indices] = prob_vector[topk_indices]

    # Renormalize
    total = np.sum(filtered)
    if total > 0:
        filtered /= total

    return filtered

def copyingProbability(wealth, bm_vector, alpha=0.3, beta=1):
    n = len(wealth)  # Number of players
    probability_matrix = np.zeros((n, n))  # Initialize the probability matrix
    wealth_norm = wealth / np.sum(wealth)

    for i in range(n):
        probability_vec = np.zeros(n)  # Initialize the probability vector for player i

        for j in range(n):
            # Compute the probability for player i comparing with player j
            diff_bm = bm_vector[j].item() - bm_vector[i].item()  # Difference in final BM values
            diff_wealth = wealth_norm[j].item() - wealth_norm[i].item()  # Difference in wealth values

            # Formula for probability
            probability_vec[j] = alpha * diff_bm + beta * diff_wealth

        # Clip values in the probability vector to 0 if less than 0
        probability_vec = np.clip(probability_vec, 0, None)

        # Calculate the total: sum of vector + number of zeros in the vector
        zero_count =  np.count_nonzero(probability_vec == 0)
        probability_vec[i] = zero_count / n
        total = np.sum(probability_vec)

        # Normalize the vector based on the total
        if total > 0:  # Avoid division by zero
            probability_vec /= total

        probability_vec = topk_with_forced_index(probability_vec, i)

        # Update the probability matrix
        probability_matrix[i, :] = probability_vec

    return probability_matrix

def networkInitial(n, wealth):
    # Create graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(n))

    # Positioning (spring layout for better spacing)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 14))  # Increase figure size

    # Draw nodes with size based on wealth
    nx.draw_networkx_nodes(G, pos, node_size=wealth, node_color='white', edgecolors='black', linewidths=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title("Initial Network Graph with Node Sizes Based on Wealth")
    plt.show()

def networkMaker(n, weightEdges, performanceVector, iterationCount, alpha = 0.3, beta =1):
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    threshold = 0

    # Add edges with weights
    for i in range(n):
        for j in range(n):
            if i != j and weightEdges[i, j] > threshold:
                G.add_edge(i, j, weight=weightEdges[i, j])

    # Flatten and clip performance vector
    performanceVector = np.array(performanceVector).flatten()
    performanceVector = np.clip(performanceVector, None, 1)
    wealth_norm = wealth / np.sum(wealth)

    # Compute positions: radial layout with high-performance nodes near center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    combined_score = alpha * performanceVector + beta * wealth_norm
    combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min())
    radii = 0.5 + (1.0 - combined_score)  # radius now ranges from 0.5 to 1.5

    pos = {
        i: (radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i]))
        for i in range(n)
    }

    # Node coloring based on bm_vector
    node_colors = [
        (1 - abs(bm), 1, 1 - abs(bm)) if bm > 0 else (1, 1 - abs(bm), 1 - abs(bm))
        for bm in performanceVector
    ]  # Green for positive, red for negative

    plt.figure(figsize=(14, 14))

    # Draw nodes with custom colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=wealth, edgecolors='black', linewidths=0.5)

    # Edge properties
    edges = G.edges(data=True)
    edge_weights = np.array([d['weight'] for (u, v, d) in edges])
    edge_colors = edge_weights / edge_weights.max() if edge_weights.max() > 0 else edge_weights

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=[w * 5 for w in edge_weights],
        alpha=0.7, edge_color=edge_colors,
        edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1
    )

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Edge Weight Intensity")

    plt.title("Weighted Network Graph at t = " + str(iterationCount + 1))
    plt.axis('off')
    plt.show()

def process_time_step_n(prob_matrix, performanceVec, wealth, alpha = 0.3, beta = 1):

    system = replace_diagonal_with_minus_one(prob_matrix)

    # Creates the b vector
    diag_elements = np.diag(prob_matrix)
    RHS = diag_elements * performanceVec.ravel() * -1

    # Solves the system and applies it to prob_matrix
    v = np.linalg.solve(system, RHS)
    phase2 = np.matmul(prob_matrix, v)

    # Updates everything
    wealth = update_wealth(wealth, phase2.reshape(n, 1))
    prob_matrix = copyingProbability(wealth, phase2, alpha, beta)

    return wealth, prob_matrix
