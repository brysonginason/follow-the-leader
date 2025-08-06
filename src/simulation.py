import argparse
import numpy as np
import matplotlib.pyplot as plt
from .models import brownian_motion, update_wealth
from .network import copying_probability, process_time_step
from .visualization import network_maker, plot_influence_over_time, network_initial_visualization
from .utils import setup_logging

np.set_printoptions(linewidth=200)


def run_advanced_simulation(n=50, steps=1000, alpha=1, beta=0, 
                           wealth_range=(250, 1000), mu_range=(-0.5, 0.5), 
                           sigma_range=(0.1, 0.2), government_intervention=-0.90,
                           visualize_at=None, track_influence=True, seed=None):
    """
    Run the advanced copycat trading simulation based on sparta.ipynb implementation.
    
    Parameters:
        n (int): Number of investors/players.
        steps (int): Number of time steps to simulate.
        alpha (float): Performance weight parameter.
        beta (float): Wealth weight parameter.
        wealth_range (tuple): Range for initial wealth (low, high).
        mu_range (tuple): Range for drift coefficients (low, high).
        sigma_range (tuple): Range for volatility coefficients (low, high).
        government_intervention (float): Clipping threshold for negative performance.
        visualize_at (list): Time steps to create network visualizations.
        track_influence (bool): Whether to track influence over time.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Results containing wealth, probability matrix, influences, and parameters.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize parameters
    wealth = np.random.randint(low=wealth_range[0], high=wealth_range[1], size=n)
    mu = np.random.uniform(mu_range[0], mu_range[1], size=n)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], size=n)
    
    if visualize_at is None:
        visualize_at = {0, 24, 49, 99, 499, steps-1}
    else:
        visualize_at = set(visualize_at)
    
    influences = [] if track_influence else None
    prob_matrix = None
    
    print(f"Starting simulation: n={n}, steps={steps}, alpha={alpha}, beta={beta}")
    
    # Show initial network if requested
    if 0 in visualize_at:
        network_initial_visualization(n, wealth)
    
    for i in range(steps):
        # Generate Brownian motion for this step
        bm_vector = brownian_motion(1, n, mu, sigma)
        bm_vector = np.clip(bm_vector, government_intervention, None)
        
        if i == 0:
            # First time step
            wealth = update_wealth(wealth, bm_vector)
            prob_matrix = copying_probability(wealth, bm_vector, alpha, beta)
        else:
            # Subsequent time steps using matrix operations
            wealth, prob_matrix = process_time_step(prob_matrix, bm_vector, wealth, alpha, beta)
        
        # Track influence of most copied node
        if track_influence:
            column_sums = prob_matrix.sum(axis=0)
            max_influence = column_sums.max() / n
            influences.append(max_influence)
        
        # Print column sums at specific intervals
        if i in visualize_at and i > 0:
            column_sums = prob_matrix.sum(axis=0)
            rounded_sums = np.round(column_sums, 3)
            print(f"\nTime step {i}:")
            for idx, val in enumerate(rounded_sums):
                print(f"Node {idx}: {val}")
        
        # Create network visualizations
        if i in visualize_at:
            network_maker(n, prob_matrix, bm_vector[:, -1], i, alpha, beta, wealth)
    
    # Plot influence over time if tracked
    if track_influence and influences:
        plot_influence_over_time(influences)
    
    return {
        'wealth': wealth,
        'probability_matrix': prob_matrix,
        'influences': influences,
        'parameters': {
            'n': n, 'steps': steps, 'alpha': alpha, 'beta': beta,
            'mu': mu, 'sigma': sigma, 'wealth_range': wealth_range
        }
    }


def compare_scenarios():
    """
    Compare different alpha/beta parameter combinations as shown in sparta.ipynb.
    """
    scenarios = [
        {'alpha': 1, 'beta': 0, 'name': 'Performance Only'},
        {'alpha': 1, 'beta': 1, 'name': 'Performance + Wealth'},
        {'alpha': 1, 'beta': 2, 'name': 'Wealth Dominant'},
        {'alpha': 1, 'beta': 500, 'name': 'Extreme Wealth Focus'}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Running scenario: {scenario['name']}")
        print(f"Alpha: {scenario['alpha']}, Beta: {scenario['beta']}")
        print(f"{'='*50}")
        
        result = run_advanced_simulation(
            n=50, 
            steps=1000,
            alpha=scenario['alpha'],
            beta=scenario['beta'],
            seed=42  # Same seed for fair comparison
        )
        
        results[scenario['name']] = result
    
    return results


def analyze_drift_influence():
    """
    Analyze which investors have the highest drift coefficients.
    """
    np.random.seed(42)
    n = 50
    mu = np.random.uniform(-0.5, 0.5, size=n)
    
    drift = np.round(mu, 4)
    
    # Find max and second max
    id_max = np.argmax(drift)
    val_max = drift[id_max]
    
    masked = drift.copy()
    masked[id_max] = -np.inf
    second_idx = np.argmax(masked)
    second_val = drift[second_idx]
    
    print(f"Highest drift - Index: {id_max}, Value: {val_max}")
    print(f"Second highest drift - Index: {second_idx}, Value: {second_val}")
    print("\nAll drift coefficients:")
    for idx, val in enumerate(drift):
        print(f"Node {idx}: {val}")
    
    return drift


def main():
    """Main function to run simulations with command line arguments."""
    parser = argparse.ArgumentParser(description="Run advanced copycat trading simulation")
    parser.add_argument("--n", type=int, default=50, help="Number of investors")
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--alpha", type=float, default=1, help="Performance weight")
    parser.add_argument("--beta", type=float, default=0, help="Wealth weight")
    parser.add_argument("--compare", action="store_true", help="Run scenario comparison")
    parser.add_argument("--analyze-drift", action="store_true", help="Analyze drift coefficients")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_scenarios()
    elif args.analyze_drift:
        analyze_drift_influence()
    else:
        run_advanced_simulation(
            n=args.n,
            steps=args.steps,
            alpha=args.alpha,
            beta=args.beta,
            seed=args.seed
        )


if __name__ == "__main__":
    main()