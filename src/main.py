from simulation import run_simulation, visualize_network
import matplotlib.pyplot as plt


def main() -> None:
    """
    Main function to run the investor simulation and visualize the final network.
    
    This function sets the simulation parameters, runs the simulation, and then visualizes
    the final investor imitation network. Errors during execution are caught and printed.
    """
    try:
        # Set parameters for the simulation (or load from a config file)
        num_investors: int = 10
        time_steps: int = 50
        alpha: float = 0.01
        beta: float = 0.001

        # Run the simulation and capture the network graph and investor list
        G, investors = run_simulation(num_investors=num_investors, time_steps=time_steps, alpha=alpha, beta=beta)

        # Visualize the final investor imitation network
        visualize_network(G)
    except Exception as e:
        print(f"An error occurred during simulation: {e}")


if __name__ == '__main__':
    main()
