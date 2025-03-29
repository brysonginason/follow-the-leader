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
        self.performance = initial_capital  # Initial performance starts at capital.
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
