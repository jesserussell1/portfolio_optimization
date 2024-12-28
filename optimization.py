# optimization.py

# Import packages
import numpy as np

# Define a function to optimize
def gradient_descent(portfolio, learning_rate=0.01, num_iterations=1000):
    weights = portfolio.weights

    # Loop over the number of iterations
    for i in range(num_iterations):
        # Simulate the portfolio and get metrics
        expected_return, risk, sharpe_ratio = portfolio.simulate()

        # Compute the gradient (numerical approximation)
        grad = np.zeros_like(weights)
        epsilon = 1e-6

        # Compute the gradient for each weight
        for j in range(len(weights)):
            weights_copy = weights.copy()
            weights_copy[j] += epsilon
            portfolio.weights = weights_copy
            expected_return_new, risk_new, _ = portfolio.simulate()
            grad[j] = (expected_return_new - expected_return) / epsilon

        # Update the weights using the gradient
        weights += learning_rate * grad
        weights = weights / np.sum(weights)

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}, Sharpe Ratio: {sharpe_ratio}, Weights: {weights}")

    # Return the optimized weights
    return weights
