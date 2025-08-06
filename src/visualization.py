import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def network_initial_visualization(n, wealth, figsize=(10, 10)):
    """
    Visualize the initial network with node sizes based on wealth.
    
    Parameters:
        n (int): Number of nodes.
        wealth (numpy.ndarray): Wealth vector.
        figsize (tuple): Figure size.
    """
    # Create positions in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

    plt.figure(figsize=figsize)
    
    # Draw nodes with size based on wealth
    for i in range(n):
        x, y = pos[i]
        plt.scatter(x, y, s=wealth[i], c='white', edgecolors='black', linewidths=0.5)
        plt.text(x, y, str(i), ha='center', va='center', fontsize=12, fontweight='bold')

    plt.title("Initial Network Graph with Node Sizes Based on Wealth")
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def network_maker(n, weight_edges, performance_vector, iteration_count, alpha, beta, 
                 wealth, figsize=(14, 14), threshold=0):
    """
    Create advanced network visualization with radial positioning and performance coloring.
    
    Parameters:
        n (int): Number of nodes.
        weight_edges (numpy.ndarray): Weight matrix for edges.
        performance_vector (numpy.ndarray): Performance vector for coloring.
        iteration_count (int): Current iteration number.
        alpha (float): Performance weight parameter.
        beta (float): Wealth weight parameter.
        wealth (numpy.ndarray): Wealth vector for node sizing.
        figsize (tuple): Figure size.
        threshold (float): Edge weight threshold for display.
    """
    # Normalize vectors
    performance_vector = np.array(performance_vector).flatten()
    performance_vector = np.clip(performance_vector, None, 1)
    wealth_norm = wealth / np.sum(wealth)

    # Compute combined score for radial positioning
    combined_score = alpha * performance_vector + beta * wealth_norm
    combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min())

    # Create radial positions
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 0.5 + (1.0 - combined_score)  # High performers closer to center

    pos = {
        i: (radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i]))
        for i in range(n)
    }

    # Node properties
    NODE_SIZE_MULTIPLIER = 40000
    node_sizes = NODE_SIZE_MULTIPLIER * wealth_norm
    
    # Node coloring based on performance (green for positive, red for negative)
    node_colors = [
        (1 - abs(bm), 1, 1 - abs(bm)) if bm > 0 else (1, 1 - abs(bm), 1 - abs(bm))
        for bm in performance_vector
    ]

    # Create plot
    plt.figure(figsize=figsize)

    # Draw nodes
    for i in range(n):
        x, y = pos[i]
        plt.scatter(x, y, s=node_sizes[i], c=[node_colors[i]], 
                   edgecolors='black', linewidths=0.5)
        plt.text(x, y, str(i), ha='center', va='center', 
                fontsize=10, fontweight='bold')

    # Draw edges
    edges_drawn = []
    edge_weights = []
    
    for i in range(n):
        for j in range(n):
            if i != j and weight_edges[i, j] > threshold:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                weight = weight_edges[i, j]
                
                plt.plot([x1, x2], [y1, y2], 'b-', 
                        linewidth=weight * 5, alpha=0.7)
                edges_drawn.append((i, j))
                edge_weights.append(weight)

    # Add colorbars
    if edge_weights:
        edge_weights = np.array(edge_weights)
        norm_weights = edge_weights / edge_weights.max() if edge_weights.max() > 0 else edge_weights
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Edge Weight Intensity", 
                           fraction=0.028, pad=0.06)

    # Colorbar for node performance
    norm_perf = plt.Normalize(vmin=performance_vector.min(), vmax=performance_vector.max())
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm_perf)
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), fraction=0.03, pad=0.04)
    cbar_nodes.set_label("Node Performance")

    plt.title(f"Weighted Network Graph at t = {iteration_count + 1}")
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_influence_over_time(influences, figsize=(10, 5)):
    """
    Plot the influence of the most copied node over time.
    
    Parameters:
        influences (list): List of influence values over time.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(range(len(influences)), influences, color='blue', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Influence of Most Copied Node")
    plt.title("Influence of Dominant Node Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_wealth_distribution(wealth, figsize=(10, 6)):
    """
    Plot wealth distribution histogram.
    
    Parameters:
        wealth (numpy.ndarray): Wealth vector.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.hist(wealth, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Wealth")
    plt.ylabel("Number of Investors")
    plt.title("Wealth Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(performances_dict, figsize=(12, 8)):
    """
    Plot performance comparison across different parameter settings.
    
    Parameters:
        performances_dict (dict): Dictionary of performance arrays with labels.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    
    for label, performances in performances_dict.items():
        plt.plot(performances, label=label, linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Performance")
    plt.title("Performance Comparison Across Different Scenarios")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()