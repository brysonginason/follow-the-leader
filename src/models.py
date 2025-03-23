import numpy as np


class Investor:
    def __init__(self, id, initial_capital):
        self.id = id
        self.capital = initial_capital
        self.performance = initial_capital  # Initial performance starts at capital.
        self.history = [initial_capital]

    def update_performance(self, dt=1, mu=0, sigma=1):
        """
        Update performance using a simple Brownian motion model:
        dX = mu * dt + sigma * sqrt(dt) * N(0,1)
        """
        delta = mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        self.performance += delta
        self.history.append(self.performance)
