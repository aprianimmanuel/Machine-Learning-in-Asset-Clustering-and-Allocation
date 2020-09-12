# Machine Learning in Asset Selection and Allocation
This project is a part of Graduation Requirement of __Job Connector Data Science and Machine Learning Program__ at __Purwadhika Digital Technology School Batch 9 Jakarta__

#### -- Project Status: [Completed]

## Introduction
My name Aprian Immanuel.
- Active/Passive Investor and Trader (2017 – now)
- Material Engineer (2016 – now)
- Entrepreneur (2016 – now)
- I am here because I want to share how to be a Data Driven Investor. 
You can find me at @aprianimmanuel / immanuelaprian@gmail.com
Linkedin: https://www.linkedin.com/in/aprianimmanuel

## Project Intro/Objective
The purpose of this project is knowing how to build our long term porfolio for Investor, Fund Manager, and other stakeholders with fully concern in maintaining and calculating the risk using Machine Learning ModelThe notebooks to this project are Python based. This project is avalaible on PyPI, meaning that you can just:

```bash
pip install mlfinlab
```

### Partner
* Purwadhika Digital Technology School Jakarta
* https://www.purwadhika.com/

### Methods Used
* Data Entry, Data Cleaning.
* Exploratory Data Analysis
* Model Selection
* Machine Learning and Model Building
* Data Visualization
* Return Based Evaluation Metrics

### Technologies
* Python
* mlfinlab
* MySql
* Pandas, jupyter
* HTML, CSS
* Flask
* JavaScript
* etc. 

## Project Description
Making the right decisions and grabbing opportunities in the fast moving world of finance can make a difference to your bottom line
This is where machine learning and artificial intelligence make a tangential difference
Market and Markets study shows that artificial intelligence in financial segment will grow to over $ 7.3 billion in 2022

Non-performing assets can cause immense losses. However, if we have ML for assesment of the assets, the system delves deeper and digs to find out all relevant information
This one area where price is influenced not just by demand-supply but also by political factors, climate, company results and unforeseen calamities.
ML keeps track of all and integrates them into a predictive capability to keep us ahead of the game

mlfinlab is a library that implements portfolio optimisation method, including classical mean-variance optimisation techniques and Black Literman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.

It is **extensive** yet easily **extensible** and can be useful for both the casual investor and the serious practitioneer. Whether we are a fundamentals-oriented investor who has identified a handful of undervalued picks or an algorithmic trader who has a basket of interesting signals. mlfinlab can help us combine our alpha streams in a risk-efficient way.

## Who use of this project

- Active/Passive Investor
- Investment Manager / Fund Manager
- Portfolio Manager
- Stock Market Researcher

## a Quick Example

Here is an example on real life stock data, demonstrating how easy it is to find the long-only portfolio that maximises the Sharpe ratio (a measure of risk-adjusted returns)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vstack,array
from math import sqrt
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Import from mlfinlab
from mlfinlab.portfolio_optimization.cla import CriticalLineAlgorithm
from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
from mlfinlab.portfolio_optimization import ReturnsEstimators
%matplotlib inline

#import historical stock prices data from 2015 - 2020
BBNI = pd.read_csv('BBNI.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
CEKA = pd.read_csv('CEKA.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
DMAS = pd.read_csv('DMAS.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
DVLA = pd.read_csv('DVLA.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
ELSA = pd.read_csv('ELSA.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
EPMT = pd.read_csv('EPMT.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
INDR = pd.read_csv('INDR.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
INDS = pd.read_csv('INDS.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
IPCC = pd.read_csv('IPCC.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
LSIP = pd.read_csv('LSIP.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
MBAP = pd.read_csv('MBAP.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
MFIN = pd.read_csv('MFIN.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
MFMI = pd.read_csv('MFMI.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
MSIN = pd.read_csv('MSIN.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
NRCA = pd.read_csv('NRCA.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
PBID = pd.read_csv('PBID.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
PGLI = pd.read_csv('PGLI.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
POWR = pd.read_csv('POWR.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
PPRE = pd.read_csv('PPRE.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
PTRO = pd.read_csv('PTRO.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
SCCO = pd.read_csv('SCCO.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
TPMA = pd.read_csv('TPMA.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
VINS = pd.read_csv('VINS.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)
WSBP = pd.read_csv('WSBP.JK.csv', parse_dates=['Date'], index_col='Date', dayfirst=True, infer_datetime_format=True, keep_date_col=True)

l = [BBNI, CEKA, DMAS, DVLA, ELSA, EPMT, INDR, INDS, IPCC, LSIP, MBAP, MFIN, MFMI, MSIN, NRCA, PBID, PGLI, POWR, PPRE, PTRO, SCCO, TPMA, VINS, WSBP]
stock_prices = pd.concat(l,keys= ['BBNI', 'CEKA', 'DMAS', 'DVLA', 
                                 'ELSA', 'EPMT', 'INDR', 'INDS',
                                 'IPCC', 'LSIP', 'MBAP', 'MFIN',
                                 'MFMI', 'MSIN', 'NRCA', 'PBID',
                                 'PGLI', 'POWR', 'PPRE', 'PTRO',
                                 'SCCO', 'TPMA', 'VINS', 'WSBP'],axis=0).reset_index()
stock_prices = stock_prices.drop(['Open', 'High', 'Low', 'Close','Volume'], axis=1)
stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
stock_prices = stock_prices.set_index('Date', drop=True)
stock_prices = stock_prices.sort_index()
stock_prices = stock_prices.pivot_table('Adj Close', ['Date'], 'level_0')
stock_prices = stock_prices.dropna(axis=0)
# stock_prices = stock_prices.resample('D').sum()

# building our variance HERC portfolio
hercMV_variance = HierarchicalEqualRiskContribution()
hercMV_variance.allocate(asset_names=stock_prices.columns,
                asset_prices=stock_prices,
                optimal_num_clusters=10,
                risk_measure='variance')
herc_MV_variance = hercMV_variance.weights
herc_MV_variance.T

# plotting our optimal portfolio
y_pos = np.arange(len(hercEW_weights.columns))

plt.figure(figsize=(25,7))
plt.bar(list(hercEW_weights.columns), hercEW_weights.values[0])
plt.xticks(y_pos, rotation=45, size=10)
plt.xlabel('Assets', size=20)
plt.ylabel('Asset Weights', size=20)
plt.title('HERC Portfolio - Equal Weighting', size=20)
# plt.savefig('HERC Barplot without K-Means Clustering using Equal Weighting as Risk Measure.png')
plt.show()
```
This output the following weights:

```txt
{'DVLA': 0.148930,
'POWR': 0.106682
'INDR': 0.031678
'PBID': 0.116460
'ELSA': 0.019759
'BBNI': 0.028351
'PPRE': 0.023513
'WSBP': 0.023767
'LSIP': 0.016655
'PTRO': 0.033891
'MBAP': 0.100444
'IPCC': 0.029781
'DMAS': 0.034415
'NRCA':	0.117436
'INDS': 0.006246
'TPMA':	0.003717
'PGLI': 0.001420
'VINS':	0.000832
'CEKA':	0.002913
'MFMI':	0.000736
'MSIN':	0.016383
'SCCO':	0.107324
'EPMT':	0.012400
'MFIN':	0.016268}
```
and will have this Evaluation Metrix as following;

 No. | Metrices | Score
----|----------|-------
1 | Conditional Drawdown at Risk | 1.079410 
2 | Expected Shortfall | -0.394162 
3 | Variance at Risk | -0.252244 
4 | Sharpe Ratio | 0.942124 
5 | Probabilistic Sharpe Ratio | 0.222700
6 | Information Ratio | 0.942124
7 | Minimum Record Length | 1167.156377
8 | Bets Concentration | 0.769090

*Disclaimer: nothing about this project constitues investment advice, and the author bears no responsibility for your subsequent investment decisions.*

### An overview of classical portfolio methods

Harry Markowitz's 1952 paper is the undeniable classic, which turned portfolio optimisation from an art into a science. The key insight is that by combining assets with different expected returns and volatilities, one can decide on a mathematically optimal allocation which minimises the risk for a target return - the set of all such optimal portfolios is referred to as the **efficient portfolio**

Although much development has been made in the subject, more than half a century later, Markowitz's care ideas are still fundamentally important and see daily use in many portfolio management firms.
The main drawback of mean-variance optimisation is that the theoritical treatment requires knowledge of the expected returns and the future risk-characteristics (covariance) of the assets. Obviously, if we knew the expected returns of a stock life would be much easier, but the whole game is that stock returns are notoriously hard to covariance based on historical data - though we do lose the theoritical guarantees provided by Markowitz, the closer our estimates are to the real values; the better our portfolio will be.

### Expected Returns

- Mean Historical Returns:
    - the simplest and most common approach, which states that the expected return of each asset is equal to the mean of its historical returns
    - easily interpretable and very intituive

### Risk models

The covariance matrix encodes not just the volatility of an asset, but also how it correlated to other assets. This is important because in order to reap the benefits of diversification (and thus increase return per unit risk), the assets in the portfolio should be as the uncorrelated possible.

- Conditional Drawdown at Risk Metrics – The average of all drawdowns or cumulative losses in excess of a certain threshold
- Expected Shortfall Risk Metrics – The expected return on the portolio in the worst q% of cases
Variance at Risk Metrics – The estimator how much a set of investments might lose (with a given probability), given a normal market conditions, in a set time period.
- Sharpe Ratio – The average return earned in excess of the risk free rate per unit of volatiity or total risk
- Probabilistic Sharpe Ratio -  The probability that the estimated Sharpe Ratio exceeds a benchmark Sharpe Ratio
- Information Ratio –  The portfolio return beyond the returns of the a benchmark, usually an index, compared to the volatility of those returns
- Minimum Record Length – The time period how long  a track record should be in order to have statistical confidence that its Sharpe rato is above a given threshold
- Bets Concentration - The uniformity of returns from bets. The closer the concentration is to 0, the more uniform the distribution of returns.

## Conclusion
- Every portfolio selection and allocation method has advantage and disadvantage sides. Investor needs to make assesment for their own risk profile.
- Portfolio selection with K-Means Clustering has a better way solution in avoiding high risk yet with high return than without K-Means Clustering
- Portfolio in Cluster 1 has better sharpe ratio but still has high variance at risk and conditional drawdown at risk compared to Cluster 2 and Cluster 3.
- Cluster 3 has less risk than Cluster 1 and Cluster 2 but lower gain  than Cluster 1 and Cluster 2. 
- Cluster 2 has more moderate risk metrics than Cluster 1 but higher gain than Cluster 3
- Every portfolio selection and allocation method depends on stock price  correlation, covariance matrix and financial stats for each company. That’s why investor needs to try different approach in finding a way to diversify and calculate risk – return based portfolio

