# graphic.py

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from portfolio import Portfolio

# Initialize portfolio
portfolio = Portfolio(num_investments=3,
                      success_probabilities=np.array([0.8, 0.6, 0.7]),
                      returns_on_success=np.array([10, 20, 15]),
                      total_capital=300)

# Number of simulations
num_simulations = 1000

# Create lists to store results
portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []

# Run multiple simulations with different random weights
for _ in range(num_simulations):
    # Generate random weights that sum to 1
    weights = np.random.random(portfolio.num_investments)
    weights /= np.sum(weights)

    # Set the portfolio weights
    portfolio.weights = weights

    # Simulate the portfolio with current weights
    expected_return, risk, sharpe_ratio = portfolio.simulate()

    # Store the results
    portfolio_returns.append(expected_return)
    portfolio_risks.append(risk)
    sharpe_ratios.append(sharpe_ratio)

# Convert lists to numpy arrays
portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the risk-return tradeoff
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10)
plt.colorbar(label='Sharpe Ratio')

# Add labels and title
plt.title('Risk-Return Tradeoff with Sharpe Ratios')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')

# Show the plot
plt.show()
