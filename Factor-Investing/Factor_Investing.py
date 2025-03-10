import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Import and load data
FamaFrenchData = pd.read_csv("/workspaces/Python/Factor-Investing/FamaFrenchFactors.csv", index_col=["Date"])



# Excess portfolio returns
FamaFrenchData['Portfolio_Excess'] = FamaFrenchData['Portfolio'] - FamaFrenchData['RF']

# Cumulative returns
CumulativeReturns = ((1 + FamaFrenchData[['Portfolio', 'Portfolio_Excess']]).cumprod() - 1)
CumulativeReturns.plot()
plt.title('Cumulative returns')
plt.show()


# Covariance matrix between Portfolio Excess and Market Excess
covariance_matrix = FamaFrenchData[['Portfolio_Excess', 'Market_Excess']].cov()

# Extract co-variance and co-efficient
covariance_coefficient = covariance_matrix.iloc[0, 1 ]
print('The covariance coefficient is', covariance_coefficient)

    # result: 5.726126338154967e-05

# Benchmark variance
benchmark_variance = FamaFrenchData['Market_Excess'].var()
print('The benchmark variance is', benchmark_variance)

    # result: 5.880335088211895e-05

# Portofio Beta
portfolio_beta = covariance_coefficient / benchmark_variance
print('\nThe portfolio beta is', portfolio_beta)
print(f'For every 1% rise (fall) in the market, the portfolio is expected to rise (fall) by {portfolio_beta:.2f}%\n')

    # result: 0.9738
    # For every 1% rise (fall) in the market, the portfolio is expected to rise (fall) by 0.97%


# Calculating beta with CAPM
import statsmodels.formula.api as smf

# CAPM regression
capm_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess', data=FamaFrenchData).fit()

print('The CAPM model adjusted R-squared is', capm_model.rsquared_adj)
print(f'The portfolio movements can be explained by {capm_model.rsquared_adj * 100:.2f}% by the market movements\n')

    # result: 0.7943
    # The portfolio movements can be explained by 79.43% by the market movements


# Extract the beta
regression_beta = capm_model.params['Market_Excess']

print('The CAPM model beta is', regression_beta)

if regression_beta == portfolio_beta:
    print('Same as the beta calculated earlier\n')
else:
    print('Divergence with the beta calculated earlier\n')

    # result: 0.9738
    # Same as the beta calculated earlier


# The Fama-French 3-factor model
FamaFrench_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML', data=FamaFrenchData).fit()

print('The FamaFrench 3 factor model R-squared is', FamaFrench_model.rsquared_adj) 

if FamaFrench_model.rsquared_adj > capm_model.rsquared_adj:
    print('The 3 factors model explains more of the portfolio movements than the CAPM model\n')
else:
    print('The 3 factors model does not explain more of the portfolio movements than the CAPM model\n')

    # result: 0.8193
    # The 3 factors model explains more of the portfolio movements than the CAPM model


# P-value of the SMB factor
smb_pvalue = FamaFrench_model.pvalues['SMB']

if smb_pvalue < 0.05:
    significant_message = 'significant'
else:
    significant_message = 'not significant'

smb_coeff = FamaFrench_model.params['SMB']
print("The SMB coefficient is", smb_coeff, "and it is", significant_message)

if smb_coeff < 0 and significant_message == 'significant':
    print('The portfolio has a significant negative exposure to small-cap stocks and a positive exposure to large-cap stocks\n')
elif smb_coeff > 0 and significant_message == 'significant':
    print('The portfolio has a significant positive exposure to small-cap stocks and a negative exposure to large-cap stocks\n')
else:
    print('The results are not significant')

    # result: The SMB coefficient is -0.2621515274319262 and it is significant
    # The portfolio has a significant negative exposure to small-cap stocks and a positive exposure to large-cap stocks

# P-value of the HML factor
hml_pvalue = FamaFrench_model.pvalues['HML']

if hml_pvalue < 0.05:
    significant_message = 'significant'
else:
    significant_message = 'not significant'

hml_coeff = FamaFrench_model.params['HML']
print("The HML coefficient is", hml_coeff, "and it is", significant_message)

if smb_coeff < 0 and significant_message == 'significant':
    print('The portfolio behaves more like a growth stock portfolio than a value stock portfolio\n')
elif smb_coeff > 0 and significant_message == 'significant':
    print('The portfolio behaves more like a value stock portfolio than a growth stock portfolio\n')
else:
    print('The results are not significant')

    # result: The HML coefficient is -0.10865715035429263 and it is significant
    # The portfolio behaves more like a growth stock portfolio than a value stock portfolio


# Portfolio Alpha
portfolio_alpha = FamaFrench_model.params['Intercept']
print('The portfolio alpha is', portfolio_alpha)

    # result: 0.01832%

portfolio_alpha_annualized = ((1 + portfolio_alpha) ** 252) - 1
print('The annualized portfolio alpha is', portfolio_alpha_annualized)

if portfolio_alpha_annualized > 0:
    print(f'The portfolio outperformed the benchmark by {portfolio_alpha_annualized * 100:.2f}% annualy\n')
else:
    print(f'The portfolio underperformed the benchmark by {portfolio_alpha_annualized * 100:.2f}% annualy\n')

    # result: 4.73%
    # The portfolio outperformed the benchmark by 4.73% annually

# Fama-French 5-factor model
FamaFrench5_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML + RMW + CMA', data=FamaFrenchData).fit()

print('The FamaFrench 5 factor model R-squared is', FamaFrench5_model.rsquared_adj)

if FamaFrench5_model.rsquared_adj > FamaFrench_model.rsquared_adj:
    print('The 5 factors model explains more of the portfolio movements than the 3 factors model\n')
else:
    print('The 5 factors model does not explain more of the portfolio movements than the 3 factors model\n')

    # result: 0.8367
    # The 5 factors model explains more of the portfolio movements than the 3 factors model
