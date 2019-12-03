# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:53:14 2019

@author: hecto
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


from scipy.special import inv_boxcox
from scipy.stats import levene
from scipy.stats import norm

from functions import helpers


#from functions.forecast_models import UnivariateLSTMSearch

if __name__ == '__main__':
    print(""" 2. Exploratory Data Analysis """)
   
    
    X = pd.read_csv('./data/store_44/X44.csv')
    holidays = pd.read_csv('./data/store_44/holidays_events44.csv')
    items = pd.read_csv('./data/store_44/items44.csv')
    oil = pd.read_csv('./data/store_44/oil44.csv')
    stores = pd.read_csv('./data/store_44/stores44.csv')
    transactions = pd.read_csv('./data/store_44/transactions44.csv')
        
    print(""" 
        Time Series Plots grouped by total unit sales, items
    """)
    
    
    store_sales = X[['date','unit_sales']].groupby(['date']).sum()
    store_sales.plot(legend=False)
    plt.show()
    
    by_item = X[['date', 'item_nbr', 'unit_sales']].groupby(['item_nbr', 'date']).sum()
    by_item.unstack(level=0).plot(legend=False)
    plt.show()
    
    print('Number of unique items: ', len(X['item_nbr'].unique()))
    
    """ remove promotions and forward fill -- as we are focused on univariate analysis, pricing changes and marketing features will not be accounted for """
    
    X.loc[X['onpromotion'] == 1, 'unit_sales'] = None
    X.fillna(method='ffill', inplace=True)
    X.loc[X['onpromotion'] == 1, 'onpromotion'] = 0
    
    """ Items for analysis will be selected based on std found in volumes"""    
    
    item_analysis_res = helpers.std_item_analysis(X)
    print(item_analysis_res)

    for k in item_analysis_res.keys():
        item, _ = item_analysis_res[k]
        
        x = X.loc[X['item_nbr'] == item, ['date','unit_sales']]
        
        x.index = x['date']
        x = x.drop(['date'], axis=1)
                
        x['unit_sales'].plot()
        plt.title = k + ' - plot'
        print('SHOWING: ', k + ' - plot')
        plt.show()
        
        sns.distplot(x['unit_sales'])
        plt.title = k + ' - histogram'
        print('SHOWING: ', k + ' - histogram')
        plt.show()
        
        sns.boxplot(x['unit_sales'])
        plt.title = k + ' - boxplot'
        print('SHOWING: ', k + ' - boxplot')
        plt.show()
    
    
    """ Explore Stationarity & Normality & Outliers """
    stationarity_res = helpers.stationarity_analysis(X, item_analysis_res)
    
                
#        q99 = x["unit_sales"].quantile(0.99)
#        q1 = x["unit_sales"].quantile(0.01)
        
        # outliers = x[(x["unit_sales"] > q99) | (x["unit_sales"] < q1)]
        
    print(stationarity_res)
    stationarity_res.to_excel("./plots/stationarity_res.xlsx")
    
    """ Auto and Partial Correlations """
    
    corr_res = helpers.corr_analysis(X, item_analysis_res, confidence=.90)
    
    corr_res.to_excel('./plots/corr_res.xlsx')
  
    for k in item_analysis_res.keys():
        item, _ = item_analysis_res[k]
        x = X.loc[X['item_nbr'] == item, 'unit_sales']
        print('SHOWING: ', k + '- auto-correlation')
        plot_acf(x, lags=365)
        print('SHOWING: ', k + '- partial-auto-correlation')
        plot_pacf(x, lags=365)
        
    X.to_excel('./data/store_44/X_eda.xlsx')
    
    """ Based on EDA following params """
    
    """ 
        At the highest hierarchical level, when all items are grouped and the store sales on a daliy
        basis are plotted, the plot shows a general upward trend with some seasonal behaviour that
        repeats on a yearly basis.  This plot was used to get an overall sense of the general behaviours
        of sales at store 44.  Next, the effect of promotions and the relevance to this project are considered.
        
        In this project unit sales for promotional weeks are removed and backfilled, since we are given 
        this flag, modelling promotions will be more easily incorporated when a multivariate model is trained.  There is an onpromotion flag 
        in the data set that can be used to train a multivariate model to help predict promotions.  Since the objective
        of this project is to compare classical and deep learning models on univariate data sets keeping promotions
        will not help in identifying which model type performs better. In the conclusion, potential advancements to the models trained in this project
        are discussed and how promotions may be included.
        
        Since the objective of this project to compare how classical methods compare to deep learning
        methods on univariate time series data, the strategy in this project was to select items that differ in their
        variability to understand how classical and deep learning approaches stack up as variability changes.
        Four items where selected from the 3671 listed items in the favorita grocery data set.
        These items were chosen based on their sales volume variability.  An item that falls in the 95% of highest
        variable items, an item that falls in the lower 5% of variability, the item the median sales volume
        variability and an item that is approximates the mean variabilitiy.  For the remainder of this paper
        I will refer to the four different data sets as follows: 'highest_std(Q95)', 'lowest_std(Q5)', median_std', 
        and 'mean_std'.
        
        Another output of exploratory data analysis was identifying if any of time series data that
        will be used to train the models is stationary and/or normally distributed.  It turns out that 
        lowest_std(Q5) is stationary when it is box-cox transformed.  In all of the other cases, the data is non-stationary.
        
        In terms of normality, the two items passed any tests for normality: median_std and highest_std(Q95).  
        These two items passed tests for normality when box-cox transformations were performed.
        Beyond statistical tests, a visual inspection of mean_std suggests that this data set approximates normality,
        however it does seem skew a bit to the left.
        
        The correlation analysis focused on auto-correlation and partial auto-correlation.  The importance of this
        analysis is to identify correlations that demonstrate strong relationships and which can be used to train our models.
        The output of the auto correlation analysis suggests that for the ARIMA models to be trained there are strong
        correlations at specific lags, which can be used as parameters for the auto-regressive 'p' and moving-average 'q'.  In turn, 
        the partial auto-correlation analysis suggests there are strong correlations which can be used as parameters
        in the for P and Q.  For the deep learning model, the auto-correlation gives us a solid starting point for the
        range of history to consider when training the LSTM models.
        
    
    """

    
    
    
    
    