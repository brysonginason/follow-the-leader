n = 10 # Number of players

# Parameters for capital per person
wealth =  np.random.randint(low = 250, high = 1000, size=n)

# Paramets for drfit
low = -0.5
high = 0.5
mu = np.random.uniform(low, high, size=n)

# Vector for volatility
low = 0.1
high = 0.2
sigma = np.random.uniform(low, high, size=n)

networkInitial(n, wealth) # 0 Time step
steps = 10  # Number of time steps

for i in range(steps):
  if i == 0:
    bm_vector = brownian_motion(1, n, mu, sigma)
    bm_vector = np.clip(bm_vector, -0.90, None) # Government intervention

    wealth = update_wealth(wealth, bm_vector)
    prob_matrix = copyingProbability(wealth, bm_vector)

  else:
    bm_vector = brownian_motion(1, n, mu, sigma)
    bm_vector = np.clip(bm_vector, -0.90, None) # Government intervention
    wealth, prob_matrix = process_time_step_n(prob_matrix, bm_vector, wealth)

  networkMaker(n, prob_matrix, bm_vector, i) #1 Time step
