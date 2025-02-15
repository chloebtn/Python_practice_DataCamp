import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

USO = pd.read_csv('/workspaces/Python/Portfolio-Risk-Management/USO.csv', index_col='Date')
#print(USO.head())

cum_returns = (1 + USO).cumprod()
#print(cum_returns.head())

# cum_returns.plot()
# plt.xticks(rotation=45)
# # plt.show()

# Historical drawdown
running_max = np.maximum.accumulate(cum_returns)
running_max[running_max < 1] = 1 # Ensure the value is never less than 1
drawdown = (cum_returns)/running_max - 1
# drawdown.plot()
# plt.xticks(rotation=45)
# # plt.show()

# Historical value at risk (VaR)
StockReturns_perc = USO * 100  # Convert to percentage
# print(StockReturns_perc.head())

var_95 = np.percentile(StockReturns_perc, 5)
print('VaR 95:', var_95)

    # Result: Historical VaR 95 : -3.60% 
    # This means that we can expect with 95% confidence that our worst daily loss will not exceed 3.60%


# Historical expected shortfall (CVaR)
cvar_95 = StockReturns_perc['USO'][StockReturns_perc['USO'] <= var_95].mean()
print('CVar 95:', cvar_95)

    # Result: Historical CVaR 95 : -5.05% 
    # This means that we can expect with 95% confidence that our average daily loss will not exceed 5.05%

sorted_rets = StockReturns_perc.sort_values(by=StockReturns_perc.columns[0])

plt.hist(sorted_rets, density=True, stacked=True)
plt.axvline(var_95, color='r', linestyle='-', linewidth=2, label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(cvar_95, color='g', linestyle='-', linewidth=2, label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.show()

