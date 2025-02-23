import pandas as pd
import numpy as np

# Load the data
prices = pd.read_csv('/workspaces/Python/2008-Crisis/crisis_portfolio.csv')

prices['Date'] = pd.to_datetime(prices['Date'], format = '%d/%m/%Y')
prices.set_index(['Date'], inplace = True)



from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
import matplotlib.pyplot as plt

mean_returns = mean_historical_return(prices, frequency = 252)

plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.title('Historical Mean Returns')
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
print('\nEfficient Covariance Matrices')
e_cov = {}
for x in epochs.keys():
    sub_price = prices.loc[epochs[x]['start']:epochs[x]['end']]
    e_cov[x] = CovarianceShrinkage(sub_price).ledoit_wolf()
    print(f'\n\nEpoch: {x}')
    print(f'Start: {epochs[x]['start']}, End: {epochs[x]['end']}')
    print('\n', e_cov[x])

    # The matrices show how the portfolio's risk increased during the crisis


# returns and covariance matrix before alone
prices_before = prices.loc[epochs['before']['start']:epochs['before']['end']]
prices_before = prices_before.drop(columns=['Citibank'], errors='ignore')
returns_before = prices_before.pct_change().dropna()
returns_before = returns_before.mean()

ecov_before = CovarianceShrinkage(prices_before).ledoit_wolf()


# returns and covariance matrix during alone
prices_during = prices.loc[epochs['during']['start']:epochs['during']['end']]
prices_during = prices_during.drop(columns=['Citibank'], errors='ignore')
returns_during = prices_during.pct_change().dropna()
returns_during = returns_during.mean()

ecov_during = CovarianceShrinkage(prices_during).ledoit_wolf()


# returns and covariance matrix after alone
prices_after = prices.loc[epochs['after']['start']:epochs['after']['end']]
prices_after = prices_after.drop(columns=['Citibank'], errors='ignore')
returns_after = prices_after.pct_change().dropna()
returns_after = returns_after.mean()

ecov_after = CovarianceShrinkage(prices_after).ledoit_wolf()


# Critical Line Algorithm
efficient_portfolio_before = CLA(returns_before, ecov_before)
#print(efficient_portfolio_before.min_volatility())

efficient_portfolio_during = CLA(returns_during, ecov_during)
#print(efficient_portfolio_during.min_volatility())

efficient_portfolio_after = CLA(returns_after, ecov_after)
#print(efficient_portfolio_after.min_volatility())


# Efficient frontiers before during after together
(ret_b, vol_b, weights_b) = efficient_portfolio_before.efficient_frontier()
(ret_d, vol_d, weights_d) = efficient_portfolio_during.efficient_frontier()
(ret_a, vol_a, weights_a) = efficient_portfolio_after.efficient_frontier()
plt.scatter(vol_b, ret_b, s=4, c='g', marker='.', label='Before')
plt.scatter(vol_d, ret_d, s=4, c='r', marker='.', label='During')
plt.scatter(vol_a, ret_a, s=4, c='b', marker='.', label='After')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier Before/During/After the Crisis')
plt.legend()
plt.show()

# Efficient frontiers separated
(ret_b, vol_b, weights_b) = efficient_portfolio_before.efficient_frontier()
plt.scatter(vol_b, ret_b, s=4, c='g', marker='.', label='Before')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier Before the Crisis')
plt.legend()
plt.show()


(ret_d, vol_d, weights_d) = efficient_portfolio_during.efficient_frontier()
plt.scatter(vol_d, ret_d, s=4, c='r', marker='.', label='During')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier During the Crisis')
plt.legend()
plt.show()


(ret_a, vol_a, weights_a) = efficient_portfolio_after.efficient_frontier()
plt.scatter(vol_a, ret_a, s=4, c='b', marker='.', label='After')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier After the Crisis')
plt.legend()
plt.show()