
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro

StockPrices = pd.read_csv("/workspaces/Python/Portfolio-Risk-Management/MSFTPrices.csv", parse_dates = ["Date"])
StockPrices = StockPrices.sort_values(by="Date")

# Daily returns on adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

StockPrices['Returns'].plot()
plt.title("Microsoft Returns")
plt.show()

percent_return = StockPrices['Returns'] * 100
returns_plot = percent_return.dropna()
returns_plot.plot(kind='hist', bins=75)
plt.title("Microsoft Returns (in %)")
plt.show()

# Average return
mean_return_daily = np.mean(StockPrices['Returns'])
mean_return_annualized = ((1 + mean_return_daily)**252) - 1
print("Average annual return: ", mean_return_annualized)

# Standard deviation of returns
sigma_daily = np.std(StockPrices['Returns'])
sigma_annualized = sigma_daily * np.sqrt(252)
print("Annualized Standard deviation: ", sigma_annualized)

# Variance of returns
variance_daily = sigma_daily**2
variance_annualized = variance_daily * 252
print("Annualized Variance: ", variance_annualized)

# Skewness of returns
clean_returns = StockPrices['Returns'].dropna()
skew_returns = skew(clean_returns)
print("Skewness of returns: ", skew_returns)

# Kurtosis of returns
excess_kurtosis = kurtosis(clean_returns)
print("Excess Kurtosis of returns: ", excess_kurtosis)
fourth_moment = excess_kurtosis + 3
print("Fourth Moment: ", fourth_moment)

# Shapiro-Wilk test for normality
shapiro_results = shapiro(clean_returns)
print(shapiro_results)
print("P-value: ", shapiro_results[1]) # p-value
if shapiro_results[1] <= 0.05:
    print("The returns are not normally distributed")
else:
    print("The returns are normally distributed")
