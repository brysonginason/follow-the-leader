import matplotlib.pyplot as plt
import networkx as nx


def plot_investor_performance(investors, figsize=(10, 6), show_plot=False):
    """
    Plot the performance history of each investor.

    Parameters:
        investors (list): List of Investor objects with 'history' and 'id' attributes.
        figsize (tuple): Figure size for the plot (default: (10, 6)).
        show_plot (bool): If True, displays the plot immediately (default: False).

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for investor in investors:
        try:
            history = investor.history
            investor_id = investor.id
        except AttributeError as e:
            raise ValueError("Each investor must have 'history' and 'id' attributes.") from e
        ax.plot(history, label=f'Investor {investor_id}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Performance')
    ax.set_title('Investor Performance Over Time')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_network(G, title="Investor Imitation Network", node_size=500, edge_width_multiplier=5, figsize=(8, 6), show_plot=False):
    """
    Visualize the investor network.

    Nodes are colored according to their current performance.
    Edge widths are proportional to the imitation weight.

    Parameters:
        G (networkx.DiGraph): The directed network graph with nodes having a 'performance' attribute and edges having a 'weight'.
        title (str): Title for the plot.
        node_size (int): Size of the nodes (default: 500).
        edge_width_multiplier (float): Multiplier to scale edge widths (default: 5).
        figsize (tuple): Figure size for the plot (default: (8, 6)).
        show_plot (bool): If True, displays the plot immediately (default: False).

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object.
    """
    pos = nx.spring_layout(G, seed=42)
    node_values = [G.nodes[n].get('performance', 0) for n in G.nodes()]
    fig, ax = plt.subplots(figsize=figsize)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_values, cmap=plt.cm.coolwarm,
        vmin=min(node_values) if node_values else 0,
        vmax=max(node_values) if node_values else 1,
        node_size=node_size, edgecolors='black'
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    edge_weights = [d.get('weight', 1) for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=[w * edge_width_multiplier for w in edge_weights], alpha=0.7,
        edge_color='gray'
    )
    ax.set_title(title)
    fig.colorbar(nodes, ax=ax, label='Investor Performance')
    ax.axis('off')
    fig.tight_layout()
    return fig, ax
