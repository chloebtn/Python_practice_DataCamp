import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

portfolio = pd.read_csv('/workspaces/Python/Portfolio-Risk-Management/CSV/crisis_portfolio.csv', index_col='Date')
portfolio.index = pd.to_datetime(portfolio.index)

# Select portfolio asset prices from the middle of the crisis
asset_prices = portfolio.loc['2008-01-01':'2009-12-31']

asset_prices.plot().set_ylabel('Closing Prices, USD')
plt.show()

# Equally weighted portfolio daily returns
asset_returns = asset_prices.pct_change()
weights = [0.25, 0.25, 0.25, 0.25]  # set equal weights
portfolio_returns = asset_returns.dot(weights)

portfolio_returns.plot().set_ylabel('Daily Return, %')
plt.show()



# Covariance matrix
covariance = asset_returns.cov() * 252
print(covariance)

    # note: Citibank has the highest annualized volatility over the time period 2008-2009

portfolio_variance = np.transpose(weights) @ covariance @ weights
portfolio_volatility = np.sqrt(portfolio_variance)
print('Portfolio volatility: ', portfolio_volatility)

    # result: 0.8475

# 30 days rolling window of portfolio returns
returns_windowed = portfolio_returns.rolling(30)

# annualize the volatility series
volatility_series = returns_windowed.std()*np.sqrt(252)

volatility_series.plot().set_ylabel ('Annualized Volatility, 30-day Window')
plt.show()



# Risk factor correlation

# Load 90-day mortgage delinquency rate
mort_del = pd.read_csv('/workspaces/Python/Portfolio-Risk-Management/CSV/mortgage_delinquency.csv', index_col='Date')
mort_del.index = pd.to_datetime(mort_del.index)

# reset portfolio_returns
returns = portfolio.pct_change()
portfolio_returns = returns.dot(weights)

# Daily portfolio returns into quaterly average returns
portfolio_q_average = portfolio_returns.resample('Q').mean().dropna()

# Daily portfolio returns into quaterly minimum returns
portfolio_q_min = portfolio_returns.resample('Q').min().dropna()

# Scatter plot between delinquency and quaterly average returns
# Subplot 1
fig, plot_average = plt.subplots()
plot_average.set_xlabel('Delinquency Rate, decimal %')
plot_average.set_ylabel('Quaterly Average Return')
# Subplot 2
fig, plot_min = plt.subplots()
plot_min.set_xlabel('Delinquency Rate, decimal %')
plot_min.set_ylabel('Quaterly Average Return')

plot_average.scatter(mort_del, portfolio_q_average)
plot_min.scatter(mort_del, portfolio_q_min)
plt.show()

    # result: It seems there is a little correlation between average returns and mordgage delinquency,
    # a stronger negative correlation exists between minimum returns and delinquency


# Regression to evaluate said correlation
import statsmodels.api as sm 

# add constant 
mort_del = sm.add_constant(mort_del)

# regression factor model fitted to the data
results_average = sm.OLS(portfolio_q_average, mort_del).fit()

# print(results_average.summary())  # do one at a time by converting one into comment

results_min = sm.OLS(portfolio_q_min, mort_del).fit()

#print(results_min.summary())    # do one at a time by converting one into comment

# quaterly portfolio average volatility
vol_wind = portfolio_returns.rolling(30).std().dropna()
vol_q_mean = vol_wind.resample('Q').mean()

results_vol = sm.OLS(vol_q_mean, mort_del).fit()

print(results_vol.summary())

    # result: Mortgage delinquency are a systematic risk factor for minimum quaterly returns and average volatily of returns
    # but not for average quaterly returns


