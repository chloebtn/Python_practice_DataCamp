
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro

# Import Data and make sure it is sorted by date
StockPrices = pd.read_csv("/workspaces/Python/Portfolio-Risk-Management/CSV/MSFTPrices.csv", parse_dates = ["Date"])
StockPrices = StockPrices.sort_values(by="Date")



# Daily returns on adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Plot returns over time
StockPrices['Returns'].plot()
plt.title("Microsoft Returns")
plt.show()

# Convert decimal returns into percentage returns
percent_return = StockPrices['Returns'] * 100
returns_plot = percent_return.dropna()  # drop missing values

# Plot the returns histogram
returns_plot.plot(kind='hist', bins=75)
plt.title("Daily Returns Distribution")
plt.show()


# Average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])

    # result: average return of the stock is 0.04% per day

# Implied annualized average return
mean_return_annualized = ((1 + mean_return_daily)**252) - 1
print("Average annual return: ", mean_return_annualized)

    # result: this implies a return of 9.99% per year



# Standard deviation of daily returns
sigma_daily = np.std(StockPrices['Returns'])

    # result: the average volatility of the stock is 1.93% per day

# Annualized standard deviation
sigma_annualized = sigma_daily * np.sqrt(252)
print("Annualized Standard deviation: ", sigma_annualized)

    # result: this implies a volatility of 30.7% per year

# Daily variance
variance_daily = sigma_daily**2

    # result: the average daily variance of the stock is 0.04% per day

# Annualized variance
variance_annualized = variance_daily * 252
print("Annualized Variance: ", variance_annualized)

    # result: annualized variance would be 9.43% per year



# Skewness of returns
clean_returns = StockPrices['Returns'].dropna()  # drop missing values
skew_returns = skew(clean_returns)  
print("Third Moment (skewness): ", skew_returns)

    # result: Skewness of the stock returns is 0.22. 
    # A normal distribution would have a skewness closer to 0, the returns are not normally distributed

# Kurtosis of returns
excess_kurtosis = kurtosis(clean_returns)   #use previously cleaned returns (in skewness calculation)
print("Excess Kurtosis of returns: ", excess_kurtosis)
fourth_moment = excess_kurtosis + 3
print("Fourth Moment (kurtosis): ", fourth_moment)

    # result: Kurtosis of the stock returns is 13.31 with an excess kurtosis of 10.31
    # A normal distribution would tend to have a kurtosis closer to 3, the returns are not normally distributed
    # note: the kurtosis function in python actually compute the excess kurtosis, to get the true fourth moment (kurtosis) we need to add 3


# Shapiro-Wilk test for normality
shapiro_results = shapiro(clean_returns)    #use previously cleaned returns (in skewness calculation)
print(shapiro_results)
print("P-value: ", shapiro_results[1])  # p-value
if shapiro_results[1] <= 0.05:
    print("The returns are not normally distributed")
else:
    print("The returns are normally distributed")

    # result: The p-value is 0, null hypothesis of normality is rejected, the data are non-normal
