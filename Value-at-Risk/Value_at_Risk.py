import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
USO = pd.read_csv('/workspaces/Python/Value-at-Risk/USO.csv', index_col='Date')

# Cumulative returns
cum_returns = (1 + USO).cumprod()
cum_returns.plot()
plt.xticks(rotation=45)
plt.title('USO Cumulative returns')
plt.show()



# Historical drawdown
# Running maximum
running_max = np.maximum.accumulate(cum_returns)
running_max[running_max < 1] = 1 # Ensure the value is never less than 1
drawdown = (cum_returns)/running_max - 1
drawdown.plot()
plt.xticks(rotation=45)
plt.title('Historical drawdown')
plt.show()

# Historical value at risk (VaR)
StockReturns_perc = USO * 100  # Convert cum_returns to percentage

#Historical VaR 95
var_95 = np.percentile(StockReturns_perc, 5)
print('VaR 95:', var_95)

    # Result: Historical VaR 95 = -3.61% 
    # In 95% of cases, the worst daily loss will not exceed 3.61%


# Historical expected shortfall (CVaR 95)
cvar_95 = StockReturns_perc['USO'][StockReturns_perc['USO'] <= var_95].mean()
print('CVar 95:', cvar_95)

    # Result: Historical CVaR 95 = -5.05% 
    # In the worst 5% of cases (when losses do exceed 3.61%), the average loss is 5.05%

# For VaR 90
var_90 = np.percentile(StockReturns_perc, 10)
print('VaR 90:', var_90)

    # Result: Historical VaR 90 = -2.56% 
    # In 90% of cases, the worst daily loss will not exceed 2.56%


# Historical expected shortfall (CVaR 90)
cvar_90 = StockReturns_perc['USO'][StockReturns_perc['USO'] <= var_90].mean()
print('CVar 90:', cvar_90)

    # Result: Historical CVaR 90 = -4.04% 
    # In the worst 10% of cases (when losses do exceed 2.56%), the average loss is 4.04%

sorted_rets = StockReturns_perc.sort_values(by=StockReturns_perc.columns[0])

plt.hist(sorted_rets, density=True, stacked=True)
plt.axvline(var_95, color='r', linestyle='-', linewidth=2, label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(cvar_95, color='r', linestyle='--', linewidth=2, label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.axvline(var_90, color='g', linestyle='-', linewidth=2, label='VaR 90: {0:.2f}%'.format(var_90))
plt.axvline(cvar_90, color='g', linestyle='--', linewidth=2, label='CVaR 90: {0:.2f}%'.format(cvar_90))
plt.legend()
plt.title('Historical Value at Risk')
plt.show()



# Parametric VaR
from scipy.stats import norm

# Store the returns in a variable for easier manipulation
StockReturns = USO['USO']

# Average daily return
mu = np.mean(StockReturns)

# Daily volatility
vol = np.std(StockReturns)

# VaR confidence level
confidence_level = 0.05

# Parametric VaR
pVaR_95 = norm.ppf(confidence_level, mu, vol)

print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(pVaR_95))

    # result: 
    # Mean:  -0.00028638956240214787 
    # Volatility:  0.021888087129708852 
    # VaR(95):  -0.03628908906473361 = -3.63%



# Scaling risk estimates (1-day to 100-days)
forecasted_values = np.empty([100, 2]) 

for i in range(100):
    forecasted_values[i, 0] = i + 1 # Time horizon (1 to 100 days)
    forecasted_values[i, 1] = abs(pVaR_95) * np.sqrt(i + 1) # Scaling the VaR

plt.plot(forecasted_values[:, 0], forecasted_values[:, 1], label='VaR')
plt.xlabel('Time Horizon (days)')
plt.ylabel('Forecasted VaR(95)')
plt.title('VaR 95 Scaled by Time')
plt.show()



# Random walk simulation

# mu and vol defined above
T = 252     # business days in a year
SO = 10     # initial stock price

# Add one to the random returns
rand_rets = np.random.normal(mu, vol, T) + 1

# Forecasted random walk
rwalk_forecast = SO * rand_rets.cumprod()

plt.plot(range(0, T), rwalk_forecast)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Random Walk Simulation')
plt.show()



# Monte Carlo simulation
np.random.seed(8)
# Loop through 100 simulations
for i in range(0, 100):
    rand_rets = np.random.normal(mu, vol, T) + 1 
    rwalk_forecast = SO * rand_rets.cumprod()
    plt.plot(range(T), rwalk_forecast)

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Monte Carlo Simulation')
plt.show()

# Monte Carlo VaR
sim_returns = []

# Loop through 100 simulations
for i in range(0, 100):
    rand_rets = np.random.normal(mu, vol, T)
    sim_returns.append(rand_rets)

# VaR(99)
var_99 = np.percentile(sim_returns, 1)
print('VaR 99: ', round(var_99 * 100, 2), '%')

    # result: -5.08%
    
