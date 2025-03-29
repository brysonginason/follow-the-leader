from simulation import run_simulation
import matplotlib.pyplot as plt


def main():
    # You can set parameters here or load them from a config file
    num_investors = 10
    time_steps = 50
    alpha = 0.01
    beta = 0.001

    run_simulation(num_investors=num_investors, time_steps=time_steps, alpha=alpha, beta=beta)
    # Plotting the results
    plt.show()


if __name__ == '__main__':
    main()
