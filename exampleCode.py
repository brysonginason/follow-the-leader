import numpy as np

# Simulation parameters
n = 50               # Number of players
alpha = 0.3          # Performance weight parameter
beta = 1             # Wealth weight parameter
steps = 10           # Number of time steps

# Parameters for capital per person
wealth = np.random.randint(low=250, high=1000, size=n)

# Parameters for drift
low, high = -0.5, 0.5
mu = np.random.uniform(low, high, size=n)

# Parameters for volatility
low, high = 0.1, 0.2
sigma = np.random.uniform(low, high, size=n)

# Initialize network at time 0
networkInitial(n, wealth)

for i in range(steps):
    # 1) simulate returns
    bm_vector = brownian_motion(1, n, mu, sigma)
    bm_vector = np.clip(bm_vector, -0.90, None)  # government intervention floor

    # 2) update wealth & copying probabilities
    if i == 0:
        wealth = update_wealth(wealth, bm_vector)
        prob_matrix = copyingProbability(wealth, bm_vector, alpha, beta)
    else:
        wealth, prob_matrix = process_time_step_n(
            prob_matrix, bm_vector, wealth, alpha, beta
        )

    # 3) evolve network
    networkMaker(n, prob_matrix, bm_vector, i, alpha, beta)
