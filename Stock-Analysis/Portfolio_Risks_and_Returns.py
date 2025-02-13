import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
stock_prices_df = pd.read_csv('/workspaces/Python/Stock-Analysis/faang_stocks.csv', index_col='Date')

# Changing the index to a datetime type allowing easier filtering and plotting
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

print(stock_prices_df)

# Plotting the stock prices
stock_prices_df.plot(title='FAANG stock prices from years 2020-2023')
plt.show()