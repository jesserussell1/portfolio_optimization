from portfolio import Portfolio
from optimization import gradient_descent
import numpy as np

# Initialize portfolio
portfolio = Portfolio(num_investments=3,
                      success_probabilities=np.array([0.8, 0.6, 0.7]),
                      returns_on_success=np.array([10, 20, 15]),
                      total_capital=300)

# Optimize portfolio using gradient descent
optimal_weights = gradient_descent(portfolio)

# Update portfolio with optimal weights
portfolio.weights = optimal_weights

# Simulate portfolio with optimal weights
expected_return, risk, sharpe_ratio = portfolio.simulate()

# Output results
print("\nOptimal Weights:", optimal_weights)
print("Best Sharpe Ratio:", sharpe_ratio)
print("Expected Portfolio Return:", expected_return)
print("Portfolio Risk (Std Dev):", risk)
