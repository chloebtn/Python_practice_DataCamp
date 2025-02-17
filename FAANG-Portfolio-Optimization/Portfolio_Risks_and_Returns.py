import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
stock_prices_df = pd.read_csv('/workspaces/Python/FAANG-Portfolio-Optimization/faang_stocks.csv', index_col='Date')

# Changing the index to a datetime type allowing easier filtering and plotting
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

print(stock_prices_df)

# Plotting the stock prices
stock_prices_df.plot(title='FAANG stock prices from years 2020-2023')
plt.show()

# Portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

# Set risk free rate
rf_rate = 0.02

# Calculate the returns and sharpe ratio for an equally weighted portfolio
returns = stock_prices_df.pct_change().dropna()
pf_weights = 5*[0.2]
benchmark_returns = returns.dot(pf_weights)
benchmark_exp_return = benchmark_returns.mean()
benchmark_volatility = benchmark_returns.std()

benchmark_sharpe_ratio = ((benchmark_exp_return - rf_rate) / benchmark_volatility) * np.sqrt(252)

# Get the optimized portfolio for minimum volatility
mu = expected_returns.mean_historical_return(stock_prices_df, compounding=False) 
s = risk_models.sample_cov(stock_prices_df) 

ef = EfficientFrontier(mu, s)

weights = ef.min_volatility()
mv_portfolio = pd.Series(weights)

mv_portfolio_vol = ef.portfolio_performance(risk_free_rate=rf_rate)[1]

# Get the optimized portfolio for the maximum sharpe ratio
ef = EfficientFrontier(mu, s)

weights = ef.max_sharpe(risk_free_rate=rf_rate)
ms_portfolio = pd.Series(weights)

ms_portfolio_sharpe = ef.portfolio_performance(risk_free_rate=rf_rate)[2]

# Get portfolios returns data
mv_portfolio_returns = returns.dot(mv_portfolio)
ms_portfolio_returns = returns.dot(ms_portfolio)

portfolios_df = pd.DataFrame({
    'Benchmark returns': benchmark_returns,
    'Min Volatility returns': mv_portfolio_returns,
    'Max Sharpe returns': ms_portfolio_returns
})

print(portfolios_df)

# PLot returns

benchmark_cumul_returns = (1 + benchmark_returns).cumprod() - 1
mv_cumulative_returns = (1 + mv_portfolio_returns).cumprod() - 1
ms_cumulative_returns = (1 + ms_portfolio_returns).cumprod() - 1

plt.plot(benchmark_cumul_returns, label='Benchmark')
plt.plot(mv_cumulative_returns, label='Min Volatility Portfolio')
plt.plot(ms_cumulative_returns, label='Max Sharpe Ratio Portfolio')
plt.title('Optimized Portfolios Returns VS Benchmark')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# Portfolios stocks and weights
print(f'Max Sharpe Ratios Portfolio stocks and respective weights:\n{mv_portfolio[mv_portfolio != 0]}')
print(f'\nMinimum Volatility Portfolio stocks and respective weights:\n{ms_portfolio[ms_portfolio != 0]}')
