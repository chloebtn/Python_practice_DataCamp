import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
StockReturns = pd.read_csv("/workspaces/Python/Big9-Portfolio-Optimization/Big9Returns2017.csv", index_col=["Date"])

# Set portfolio weights
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Portfolio return
StockReturns["Portfolio"] = WeightedReturns.sum(axis=1)

# Cumulative Portfolio Returns for future use in plotting
CumulativeReturns = ((1 + StockReturns["Portfolio"]).cumprod() - 1)



# Set number of stocks in portfolio (will be used many times)
numstocks = 9   



# Equally weighted portfolio
# Array of equal weights
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)  

# Equally Weighted Portfolio Returns
StockReturns["Portfolio_EW"] = StockReturns.iloc[:, 0:numstocks].mul(portfolio_weights_ew, axis=1).sum(axis=1)

# Cumulative Portfolio Returns for future use in plotting
CumulativeReturns_EW = ((1 + StockReturns["Portfolio_EW"]).cumprod() - 1)



# Market-cap weighted portfolio
# Array of market capitalizations in billions
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# Market cap weights
mcap_weights = market_capitalizations / np.sum(market_capitalizations)

# Market cap weighted portfolio returns
StockReturns["Portfolio_MCap"] = StockReturns.iloc[:, 0:numstocks].mul(mcap_weights, axis=1).sum(axis=1)

# Cumulative Portfolio Returns for future use in plotting
CumulativeReturns_MCap = ((1 + StockReturns["Portfolio_MCap"]).cumprod() - 1)



# Setting up the data for Max Sharpe Ratio and Global Minimum Variance Portfolios
# The Correlation Matrix
correlation_matrix = StockReturns.iloc[:, 0:numstocks].corr()   # taking only stock columns

# Heatmap of the correlation matrix
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=0.3, annot_kws={"size": 8})
plt.gca().xaxis.tick_top()
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

    # note: Amazon is most correlated with Facebook and Microsoft


# The Covariance Matrix
cov_mat = StockReturns.iloc[:, 0:numstocks].cov()
cov_mat_annual = cov_mat * 252


# Portfolio standard deviation (from first portfolio above)
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
#print('Portfolio volatility:', portfolio_volatility)

    # result: 8.93%



# Markowitz Portfolios

# Create random portfolios
# Set seed for reproducibility
np.random.seed(42)

# Extract only stock columns
stock_columns = StockReturns.columns[0:numstocks]

# Historical mean returns (+ covariance matrix which we already have in cov_mat)
mean_annual_returns = StockReturns[stock_columns].mean() * 252

# Number of portfolios generated
num_portfolios = 10000

# Storage for portfolio data
rd_weights = []
rd_returns = []
rd_volatilities = []

# Generate the portfolios
for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(numstocks) * 0.5) # generate random weights
    port_return = np.dot(weights, mean_annual_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat_annual, weights)))

    # Append data
    rd_weights.append(weights)
    rd_returns.append(port_return)
    rd_volatilities.append(port_volatility)

# Create DataFrame for random portfolios
random_portfolios = pd.DataFrame(rd_weights, columns=stock_columns)
random_portfolios['Returns'] = rd_returns
random_portfolios['Volatility'] = rd_volatilities

# Add the Sharpe Ratio
rf = 0      # risk-free rate

random_portfolios['Sharpe'] = (random_portfolios['Returns'] - rf) / random_portfolios['Volatility']

print(f'Historical Sharpe ratios range from {random_portfolios['Sharpe'].describe()['min']:.2f} to {random_portfolios['Sharpe'].describe()['max']:.2f}')

    # result: Historical Sharpe ratios range from -2.51 to 3.45



# Max Sharpe Ratio portfolio
sorted_portfolios_s = random_portfolios.sort_values(by=['Sharpe'], ascending=False) # sort by sharpe ratios

MSR_weights = sorted_portfolios_s.iloc[0, 0:numstocks]
MSR_weights_array = np.array(MSR_weights)

# MSR Portfolio Returns
StockReturns["Portfolio_MSR"] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)

# Cumulative returns for future use
CumulativeReturns_MSR = ((1 + StockReturns["Portfolio_MSR"]).cumprod() - 1)



# Global Minimum Variance Portfolio
sorted_portfolios_v = random_portfolios.sort_values(by=['Volatility'], ascending=True)  # sort by volatilities

GMV_weights = sorted_portfolios_v.iloc[0, 0:numstocks]
GMV_weights_array = np.array(GMV_weights)

# GMV Portfolio Returns
StockReturns["Portfolio_GMV"] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

# Cumulative returns for future use
CumulativeReturns_GMV = ((1 + StockReturns["Portfolio_GMV"]).cumprod() - 1)



# Plot the different portfolios
CumulativeReturns_EW.plot()
CumulativeReturns_MCap.plot()
CumulativeReturns_MSR.plot()
CumulativeReturns_GMV.plot()
plt.legend()
plt.title('Performance of the different portfolios')
plt.show()
