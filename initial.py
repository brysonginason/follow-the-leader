import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)


# Creates an n-size vector of random integers btn low and high that sum to S.
def random_partition(S, n, low, high):
    """
    Parameters:
        S (int): Target sum.
        n (int): Number of elements.
        low (int): Minimum value per element
    """
    # Generate n random numbers
    values = np.random.randint(low, high, size=n)

    # Scale them to ensure they sum to S
    values = (values / values.sum()) * S
    values = np.round(values).astype(int)  # Convert to integers

    # Adjust rounding errors to ensure exact sum
    diff = S - values.sum()
    values[np.random.choice(n)] += diff  # Add/subtract the difference to a random element

    return values


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
    mu = np.asarray(mu).reshape(n, 1)     # Ensure mu is (n,1) for broadcasting
    sigma = np.asarray(sigma).reshape(n, 1)  # Ensure sigma is (n,1) for broadcasting

    dW = np.sqrt(dt) * np.random.randn(n,steps)  # Standard Brownian increments
    W = np.cumsum(sigma * dW, axis=1)  # Scale by volatility and integrate
    t = np.arange(1, steps + 1) * dt  # Time steps
    drift = mu * t  # Linear drift component

    return W + drift  # Add drift


# Update wealth based on Brownian motion.
def update_wealth(wealth, bm_vector):
    """
    Parameters:
        wealth (numpy.ndarray): Vector of wealth for each player.
        bm_vector (numpy.ndarray): Brownian motion vector (n x steps).
    """
    return wealth * (1 + bm_vector.T)


def copyingProbability(wealth, bm_vector, alpha=1, beta=1):
    n = len(wealth)  # Number of players
    probability_matrix = np.zeros((n, n))  # Initialize the probability matrix
    wealth_norm = wealth / np.sum(wealth)

    for i in range(n):
        probability_vec = np.zeros(n)  # Initialize the probability vector for player i

        for j in range(n):
            # Compute the probability for player i comparing with player j
            diff_bm = bm_vector[j] - bm_vector[i]  # Diff. in final BM values
            diff_wealth = wealth_norm[j] - wealth_norm[i]  # Difference in wealth values

            # Formula for probability
            probability_vec[j] = alpha * diff_bm + beta * diff_wealth

        # Clip values in the probability vector to 0 if less than 0
        probability_vec = np.clip(probability_vec, 0, None)

        # Calculate the total: sum of vector + number of zeros in the vector
        zero_count = np.count_nonzero(probability_vec == 0)
        total = np.sum(probability_vec) + zero_count

        # Normalize the vector based on the total
        if total > 0:  # Avoid division by zero
            probability_vec /= total
            probability_vec[i] = zero_count / total

        # Update the probability matrix
        probability_matrix[i, :] = probability_vec

    return probability_matrix
