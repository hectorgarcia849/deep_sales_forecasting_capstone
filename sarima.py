# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:41:03 2019

@author: hecto
"""

import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

from functions.forecast_models import SarimaxSearch
#from functions.forecast_models import UnivariateLSTMSearch

import time



STORENUM = 44

# new plan -- only work with store 44 data, create SARIMAX models for each product

if __name__ == '__main__':
    """ Based on out come of EDA items to be used for model building:
        
        'highest_std(Q95)': (1463797, 42.26034314183367),
        'lowest_std(Q5)': (1230461, 1.563456760755706),
        'median_std': (580929, 7.265907559138344),
        'mean_std': (111397, 14.388653833110155)}
      
    """
    LAST_TWO_YEARS = -730
    X = pd.read_excel('./data/store_44/X_eda.xlsx')
    X = X.loc[(X['item_nbr'] == 1463797) | (X['item_nbr'] == 1230461) | (X['item_nbr'] == 580929) | (X['item_nbr'] == 111397), :]
    
    """ Split and Standardize data for model training """
    scaler = StandardScaler()
    scaler.fit(X['unit_sales'].values.reshape(-1, 1))
    X['unit_sales_std'] = scaler.transform(X['unit_sales'].values.reshape(-1, 1)) 
    
    train = X[:int(len(X)*.8)-1]
    test = X[int(len(X)*.8):]
    
    print('train: ', train)
    
    """ Params dictionary for grid search based on output of EDA """
    params_dict = {}
    params_dict['highest_std(Q95)'] = {'item_nbr': 1463797, 'p': [7], 'd':[1], 'q':[7], 'P': [0, 7], 'D': [0], 'Q': [0, 7] }
#    params_dict['lowest_std(Q5)'] = {'item_nbr': 1230461, 'p': [0], 'd':[0,1], 'q': [0,1], 'P': [0], 'D': [0], 'Q': [0]}
    params_dict['median_std'] = {'item_nbr': 580929,'p': [14] , 'd': [1], 'q':  [0], 'P': [0, 7] , 'D': [0], 'Q': [0, 7]}
    params_dict['mean_std'] = {'item_nbr': 111397,'p': [0], 'd':[0, 1], 'q': [0,1], 'P': [0], 'D': [0], 'Q': [0]}
    
    print('starting timer...')
    then = time.time()
    
    models_fit = {}
    
    for k in params_dict.keys():
        print('Model Building: ', k)
        item = params_dict[k]['item_nbr']
        history = train.loc[train['item_nbr'] == item, 'unit_sales_std'].values.tolist()
        print('item: ', item, 'n: ', history)
        sarima_search = SarimaxSearch(history, params_dict[k])
        scores = sarima_search.grid_search(True)
        models_fit[item] = scores
        
        for config, error, model, history in scores:
            print(config, error)
            
    now = time.time() #Time after it finished

    print("It took: ", now-then, " seconds")
    
    results = {}
    
    for k in params_dict.keys():
        item = params_dict[k]['item_nbr']
        results[k] = {}
        print('starting test for item: ', item)
        X = test.loc[test['item_nbr'] == item, 'unit_sales_std'].shift(1).dropna().values.tolist()[:-1]
        y = test.loc[test['item_nbr'] == item, 'unit_sales_std'].dropna().values.tolist()[1:]
        
        top = 0
        pred = list()
        actual = list()
        print("Right before error: ", models_fit)
        config, _, model_fit, history = models_fit[item][top]
        
        for t in range(len(y)):
            order, sorder, trend = config
            model = SARIMAX(history, order=order, sorder=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            pred.append(yhat)
            actual.append(y[t])
            history.append(y[t])
    
        rmse = SarimaxSearch.measure_rmse(actual, pred)
        mae = SarimaxSearch.measure_mae(actual, pred)
        error = {'rmse': rmse, 'mae': mae, 'avg_rmse_mae': (rmse + mae) / 2}
        results[item] = error
        print('Test, mae: ', error['mae'])
        
    print(results)
    
    outputfile = open('results', 'ab')
    pickle.dump(results, outputfile)
    outputfile.close()
    
    
    
