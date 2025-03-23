import matplotlib.pyplot as plt
import networkx as nx


def plot_investor_performance(investors):
    """
    Plot the performance history of each investor.

    Parameters:
        investors (list): List of Investor objects with a 'history' attribute.
    """
    plt.figure(figsize=(10, 6))
    for investor in investors:
        plt.plot(investor.history, label=f'Investor {investor.id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Performance')
    plt.title('Investor Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_network(G, title="Investor Imitation Network"):
    """
    Visualize the investor network.

    Nodes are colored according to their current performance.
    Edge widths are proportional to the imitation weight.

    Parameters:
        G (networkx.DiGraph): The directed network graph.
        title (str): Title for the plot.
    """
    pos = nx.spring_layout(G, seed=42)

    # Get current performance for each node
    node_values = [G.nodes[n].get('performance', 0) for n in G.nodes()]

    # Draw nodes with a color mapping based on performance
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_values, cmap=plt.cm.coolwarm, 
        vmin=min(node_values) if node_values else 0, 
        vmax=max(node_values) if node_values else 1, 
        node_size=500, edgecolors='black'
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with widths scaled by imitation weight
    edge_weights = [d['weight'] for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos, width=[w * 5 for w in edge_weights], alpha=0.7, 
        edge_color='gray'
    )

    plt.title(title)
    plt.colorbar(nodes, label='Investor Performance')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
