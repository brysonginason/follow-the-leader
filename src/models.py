import numpy as np


class Investor:
    def __init__(self, id: int, initial_capital: float) -> None:
        """
        Initialize an Investor with an identifier and an initial capital.

        Parameters:
            id (int): Unique identifier for the investor.
            initial_capital (float): The starting capital for the investor. Must be non-negative.
        """
        if initial_capital < 0:
            raise ValueError("Initial capital must be non-negative.")
        self.id = id
        self.capital = initial_capital
        self.performance = initial_capital
        self.history = [initial_capital]

    def update_performance(self, dt: float = 1, mu: float = 0, sigma: float = 1) -> None:
        """
        Update performance using a simple Brownian motion model:
        dX = mu * dt + sigma * sqrt(dt) * N(0,1)

        Parameters:
            dt (float): Time step for the update. Must be positive.
            mu (float): Drift coefficient.
            sigma (float): Volatility coefficient.
        """
        if dt <= 0:
            raise ValueError("Time step 'dt' must be positive.")
        delta = mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        self.performance += delta
        self.history.append(self.performance)


def brownian_motion(steps, n, mu, sigma, dt=1):
    """
    Generate d independent Brownian motions of size n with drift and volatility.
    
    Parameters:
        steps (int): Number of time steps.
        n (int): Number of independent Brownian motions.
        mu (float or array): Drift term per investor.
        sigma (float or array): Volatility term per investor.
        dt (float): Time step size (default is 1).
    
    Returns:
        numpy.ndarray: Brownian motion paths (n x steps).
    """
    mu = np.asarray(mu).reshape(n, 1)
    sigma = np.asarray(sigma).reshape(n, 1)

    dW = np.sqrt(dt) * np.random.randn(n, steps)
    W = np.cumsum(sigma * dW, axis=1)
    t = np.arange(1, steps + 1) * dt
    drift = mu * t

    return W + drift


def update_wealth(wealth, bm_vector):
    """
    Update wealth based on Brownian motion vector.
    
    Parameters:
        wealth (numpy.ndarray): Vector of wealth for each player.
        bm_vector (numpy.ndarray): Brownian motion vector for this time step.
    
    Returns:
        numpy.ndarray: Updated wealth vector.
    """
    return wealth * (1 + bm_vector[:, -1])


def topk_with_forced_index(prob_vector, i):
    """
    Filter probability vector to top-k values with forced inclusion of index i.
    
    Parameters:
        prob_vector (numpy.ndarray): Probability vector to filter.
        i (int): Index to force inclusion.
    
    Returns:
        numpy.ndarray: Filtered and normalized probability vector.
    """
    n = len(prob_vector)
    k = max(5, int(np.ceil(n / 100)))

    # Get indices of top-k values
    topk_indices = np.argpartition(prob_vector, -k)[-k:]

    # Create filtered vector
    filtered = np.zeros_like(prob_vector)
    filtered[topk_indices] = prob_vector[topk_indices]

    # Renormalize
    total = np.sum(filtered)
    if total > 0:
        filtered /= total

    return filtered


def replace_diagonal_with_minus_one(matrix):
    """Replace diagonal elements with -1."""
    np.fill_diagonal(matrix, -1)
    return matrix