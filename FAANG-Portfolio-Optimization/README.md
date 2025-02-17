The goal of this exercise was to find and evaluate a set of portfolios using mean-variance optimization using libraries such as PyPortfolioOpt.


Here are some of the results:

#### Stock Prices DataFrame:
<img width="316" alt="image" src="https://github.com/user-attachments/assets/98a295e1-9939-45cc-a2c2-8c3d3b00b29b" />

#### PLot of the Stock Prices from 2020 to 2023:
![image](https://github.com/user-attachments/assets/b46a7096-8b90-4223-9d0a-025357baa772)

We observe that, while apple, amazon and google stocks seem more stable, the netflix stock went through and important fall in price in 2022 and seems a lot more volatile. 
That is not a surprising thing to observe considering the year the streaming platform had in 2022 due to more than 1,2 million subscribers leaving after price hikes, password sharing policies announcement, and increasing competition such as Amazon, Disney+ or HBO Max. Slowing growth had investors worried and led to a massive sell-off. However, big releases and comebaks of hit shows, as well as the introduction of an add-supported plan later that year brought to the service 7.7 million new users, allowing Netflix the comeback we observe at the end of 2022.

It will be interesting to see how the optimization methods dealt with that stock.

#### Returns of Benchmark (equally weighted portfolio), and optimized portfolios (Max Sharpe Ratio and Minimun Volatility):
<img width="442" alt="image" src="https://github.com/user-attachments/assets/229cff3c-db63-4b89-9e33-761da822d0b0" />

#### Plot of the optimized portfolios returns versus the benchmark:
![image](https://github.com/user-attachments/assets/4f9845f7-b0b5-4beb-a199-cce0ab7f1228)

#### Selected stocks and weights:
<img width="361" alt="image" src="https://github.com/user-attachments/assets/b2810015-d795-40a8-94c8-c27264a94483" />

It is interesting to see that the max sharpe portfolio chose four out of the five stocks, including the 3 more stable stocks apple amazon and google, and slightly balanced them with the netflix stock, which, we observed earlier, is a lot more volatile although it did get more stable after 2022, but do offer much higher returns. The meta stock, him, has been left out.

The minimum volatility portfolio chose to put more than 81% of it's weight into apple, which based on the precedent plot seems reasonable considering that it appears to be the most stable one out of the five. The Netflix stock was once again selected with 0.5% of total weight.
