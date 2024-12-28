# portfolio.py

# Import packages
import numpy as np

# Create a Portfolio class
class Portfolio:
    def __init__(self, num_investments, success_probabilities, returns_on_success, total_capital,
                 perturbation_factor=0.05):
        self.num_investments = num_investments

        # Add noise to initial parameters
        self.success_probabilities = self.add_perturbation(success_probabilities, perturbation_factor)
        self.returns_on_success = self.add_perturbation(returns_on_success, perturbation_factor)
        self.total_capital = total_capital
        self.weights = np.array([1 / num_investments] * num_investments)  # Initial equal distribution

    # Define a method to add perturbation
    def add_perturbation(self, values, perturbation_factor):
        return values * (1 + np.random.uniform(-perturbation_factor, perturbation_factor, size=values.shape))

    # Define a method to simulate the portfolio
    def simulate(self, num_simulations=10000, perturbation_factor=0.05):
        portfolio_returns = []
        for _ in range(num_simulations):
            # Add noise during each simulation
            perturbed_probabilities = self.add_perturbation(self.success_probabilities, perturbation_factor)
            perturbed_returns = self.add_perturbation(self.returns_on_success, perturbation_factor)

            # Simulate outcomes with perturbed values
            outcomes = np.random.binomial(1, perturbed_probabilities)
            returns = outcomes * perturbed_returns
            portfolio_return = np.dot(self.weights, returns)
            portfolio_returns.append(portfolio_return)

        # Calculate portfolio metrics
        portfolio_returns = np.array(portfolio_returns)
        expected_return = np.mean(portfolio_returns)
        risk = np.std(portfolio_returns)
        sharpe_ratio = expected_return / risk if risk != 0 else 0

        # Return portfolio metrics
        return expected_return, risk, sharpe_ratio
