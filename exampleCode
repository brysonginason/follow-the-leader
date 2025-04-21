n = 50 # Number of players

# Parameters for capital per person
wealth =  np.random.randint(low = 250, high = 1000, size=n)
# print("Initial total wealth: ", np.sum(wealth))

# Paramets for drfit
low = -0.5
high = 0.5
mu = np.random.uniform(low, high, size=n)

# Vector for volatility
low = 0.1
high = 0.2
sigma = np.random.uniform(low, high, size=n)

networkInitial(n, wealth) # 0 Time step
alpha = 0.3 # Performance weight parameter
beta = 1 # Wealth weight parameter
steps = 5  # Number of time steps

for i in range(steps):
  if i == 0:
    bm_vector = brownian_motion(1, n, mu, sigma)
    bm_vector = np.clip(bm_vector, -0.90, None) # Government intervention

    wealth = update_wealth(wealth, bm_vector)
    prob_matrix = copyingProbability(wealth, bm_vector, alpha , beta)
    #print(np.round(prob_matrix, 3))
    #print(np.round(bm_vector, 3))

  else:
    bm_vector = brownian_motion(1, n, mu, sigma)
    bm_vector = np.clip(bm_vector, -0.90, None) # Government intervention
    wealth, prob_matrix = process_time_step_n(prob_matrix, bm_vector, wealth, alpha , beta)
    #print(np.round(prob_matrix, 3))
    #print(np.round(bm_vector, 3))

  # print("total wealth: ", np.sum(wealth))

  networkMaker(n, prob_matrix, bm_vector, i, alpha, beta) #1 Time step
