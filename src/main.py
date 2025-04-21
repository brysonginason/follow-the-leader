from simulation import run_simulation
from visualization import plot_investor_performance, plot_network
import matplotlib.pyplot as plt


def main():
    try:
        # Simulation parameters
        num_investors = 10
        time_steps = 5
        alpha = 0.01
        beta = 0.001
        mu = 0.0         # Drift coefficient
        sigma = 5.0      # Volatility coefficient

        # Run simulation with drift and volatility parameters
        G, investors = run_simulation(num_investors, time_steps, alpha, beta, mu, sigma)

        # Plot without immediately showing
        plot_investor_performance(investors, show_plot=False)
        plot_network(G, show_plot=False)

        # Show all plots at once
        plt.show()

    except Exception as e:
        print(f"An error occurred during simulation: {e}")


if __name__ == '__main__':
    main()
