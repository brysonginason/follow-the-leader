import numpy as np
from .models import topk_with_forced_index, replace_diagonal_with_minus_one


def copying_probability(wealth, bm_vector, alpha, beta):
    """
    Compute the copying probability matrix for all investor pairs.
    
    Parameters:
        wealth (numpy.ndarray): Wealth vector for each investor.
        bm_vector (numpy.ndarray): Brownian motion vector for this time step.
        alpha (float): Performance weight parameter.
        beta (float): Wealth weight parameter.
    
    Returns:
        numpy.ndarray: Probability matrix (n x n).
    """
    n = len(wealth)
    probability_matrix = np.zeros((n, n))
    wealth_norm = wealth / np.sum(wealth)

    for i in range(n):
        probability_vec = np.zeros(n)

        for j in range(n):
            diff_bm = bm_vector[j].item() - bm_vector[i].item()
            diff_wealth = wealth_norm[j].item() - wealth_norm[i].item()
            probability_vec[j] = alpha * diff_bm + beta * diff_wealth

        # Clip negative values to 0
        probability_vec = np.clip(probability_vec, 0, None)

        # Handle self-copying probability
        zero_count = np.count_nonzero(probability_vec == 0)
        probability_vec[i] = zero_count / n
        total = np.sum(probability_vec)

        # Normalize
        if total > 0:
            probability_vec /= total

        # Apply top-k filtering
        probability_vec = topk_with_forced_index(probability_vec, i)
        probability_matrix[i, :] = probability_vec

    return probability_matrix


def process_time_step(prob_matrix, performance_vec, wealth, alpha, beta):
    """
    Process one time step of the simulation using matrix operations.
    
    Parameters:
        prob_matrix (numpy.ndarray): Current probability matrix.
        performance_vec (numpy.ndarray): Performance vector for this time step.
        wealth (numpy.ndarray): Current wealth vector.
        alpha (float): Performance weight parameter.
        beta (float): Wealth weight parameter.
    
    Returns:
        tuple: Updated (wealth, probability_matrix).
    """
    # Create system matrix
    system = replace_diagonal_with_minus_one(prob_matrix)

    # Create RHS vector
    diag_elements = np.diag(prob_matrix)
    RHS = diag_elements * performance_vec.ravel() * -1

    # Solve the system
    v = np.linalg.solve(system, RHS)
    phase2 = np.matmul(prob_matrix, v)

    # Update wealth and probability matrix
    wealth = wealth * (1 + phase2.reshape(len(wealth)))
    prob_matrix = copying_probability(wealth, phase2, alpha, beta)

    return wealth, prob_matrix