import pandas as pd
import numpy as np

# Load the data
prices = pd.read_csv('/workspaces/Python/Portfolio-Risk-Management/CSV/crisis_portfolio.csv')

prices['Date'] = pd.to_datetime(prices['Date'], format = '%d/%m/%Y')
prices.set_index(['Date'], inplace = True)



from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
import matplotlib.pyplot as plt

mean_returns = mean_historical_return(prices, frequency = 252)

plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.show()

# sample covariance matrix of returns
sample_cov = prices.pct_change().cov() * 252

# efficient covariance matrix of returns (shrink errors)
e_cov = CovarianceShrinkage(prices).ledoit_wolf()

print("Sample Covariance Matrix\n", sample_cov, "\n")
print("Efficient Covariance Matrix\n", e_cov, "\n")


# Dictionnary of time periods ('epochs)
epochs = { 'before' : {'start': '1-1-2005', 'end': '31-12-2006'},
           'during' : {'start': '1-1-2007', 'end': '31-12-2008'},
           'after'  : {'start': '1-1-2009', 'end': '31-12-2010'}
         }

# Efficient covariance for each epoch
e_cov = {}
for x in epochs.keys():
    sub_price = prices.loc[epochs[x]['start']:epochs[x]['end']]
    e_cov[x] = CovarianceShrinkage(sub_price).ledoit_wolf()

print('Efficient Covariance Matrices\n', e_cov)

    # The matrices show how the portfolio's risk increased during the crisis


# returns and covariance matrix during alone
prices_during = prices.loc[epochs['during']['start']:epochs['during']['end']]
prices_during = prices_during.drop(columns=['Citibank'], errors='ignore')
returns_during = prices_during.pct_change().dropna()
returns_during = returns_during.mean()

ecov_during = CovarianceShrinkage(prices_during).ledoit_wolf()


# Critical Line Algorithm
efficient_portfolio_during = CLA(returns_during, ecov_during)
print(efficient_portfolio_during.min_volatility())

# Efficient frontier
(ret, vol, weights) = efficient_portfolio_during.efficient_frontier()

plt.scatter(vol, ret, s=4, c='g', marker='.')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier During the Crisis')
plt.legend()
plt.show()