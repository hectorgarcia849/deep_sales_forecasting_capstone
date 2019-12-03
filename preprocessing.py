# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:47:56 2019

@author: hecto
"""

import pandas as pd
import numpy as np

from functions import helpers

if __name__ == '__main__':
    
    """ 1. IMPORT  & PREPROCESS DATA """    
    
    X = pd.read_csv('./data/train44.csv').drop(['Unnamed: 0'], axis=1)
    holidays = pd.read_csv('./data/holidays_events.csv')
    items = pd.read_csv('./data/items.csv')
    oil = pd.read_csv('./data/oil.csv')
    stores = pd.read_csv('./data/stores.csv')
    transactions = pd.read_csv('./data/transactions.csv')
    
    
    all_data = {\
                'X': {'data': X}, \
                'holidays': {'data': holidays}, \
                'items': {'data': items}, \
                'oil': {'data': oil}, \
                'stores': {'data': stores}, \
                'transactions': {'data': transactions} \
                }
    
    nan_values = helpers.get_nan_values_report(all_data)
    
    X['onpromotion'] = X['onpromotion'].fillna(False)
    X.loc[X['unit_sales'] <= 0, 'unit_sales'] = np.nan
    X['unit_sales'] = X['unit_sales'].fillna(method='ffill')
    print(X['unit_sales'].describe())
    
    
    n_train = len(X)
    n_unit_sales_gt_200 = pd.DataFrame(X[X['unit_sales'] > 200]['date'].value_counts())
    n_unit_sales_gt_200 = n_unit_sales_gt_200.reset_index(level=0)
    n_unit_sales_gt_200.columns = ['date', 'gt_200_count']
    
    cross_ref_holiday = pd.merge(n_unit_sales_gt_200, holidays[['date', 'description']], on=['date', 'date'], how='left')
    
    promos_gt_200 = X.loc[(X['onpromotion'] == 1) & (X['unit_sales'] > 200), 'date'].value_counts()
    promos_gt_200 = promos_gt_200.reset_index(level=0)
    promos_gt_200.columns = ['date', 'gt_200_promo']
    
    cross_ref_promo = pd.merge(cross_ref_holiday, promos_gt_200, on=['date', 'date'], how='left')
    
    X['onpromotion'] = X['onpromotion'].astype('uint8')
    X['date'] = pd.to_datetime(X['date'], infer_datetime_format=True)
    X = X.drop(['id'], axis=1)

    oil.loc[0, 'dcoilwtico'] = oil.loc[1, 'dcoilwtico']
    oil = oil.fillna(method='ffill')


    holidays['date'] = pd.to_datetime(holidays['date'], infer_datetime_format=True)
    holidays = pd.concat([holidays, pd.get_dummies(holidays['type'])], axis=1).drop(['type'], axis=1)
    holidays = pd.concat([holidays, pd.get_dummies(holidays['locale'])], axis=1).drop(['locale'], axis=1)
    holidays = pd.concat([holidays, pd.get_dummies(holidays['locale_name'])], axis=1).drop(['locale_name'], axis=1)
    holidays = pd.concat([holidays, pd.get_dummies(holidays['description'])], axis=1).drop(['description'], axis=1)
    holidays['transferred'] = holidays['transferred'].astype('uint8')


    items['perishable'] = items['perishable'].astype('uint8')
    items = pd.concat([items, pd.get_dummies(items['family'])], axis=1).drop(['family'], axis=1)
    items = pd.concat([items, pd.get_dummies(items['class'])], axis=1).drop(['class'], axis=1)
    items.index = items['item_nbr']
    items = items.drop(['item_nbr'], axis=1)
    
    stores = pd.concat([stores, pd.get_dummies(stores['city'])], axis=1).drop(['city'], axis=1)
    stores = pd.concat([stores, pd.get_dummies(stores['state'])], axis=1).drop(['state'], axis=1)
    stores = pd.concat([stores, pd.get_dummies(stores['type'])], axis=1).drop(['type'], axis=1)
    stores['store_nbr'] = stores['store_nbr'].astype('uint8')
    stores['cluster'] = stores['cluster'].astype('uint8')
    stores.index = stores['store_nbr']
    stores = stores.drop(['store_nbr'], axis=1)
    
    stats = helpers.get_descriptive_stats(all_data)

    transactions['date'] = pd.to_datetime(transactions['date'], infer_datetime_format=True)
    transactions['transactions'] = transactions['transactions'].astype('uint16')
        
    X.to_csv('./data/store_44/X44.csv', index=False)
    holidays.to_csv('./data/store_44/holidays_events44.csv', index=False)
    items.to_csv('./data/store_44/items44.csv', index=False)
    oil.to_csv('./data/store_44/oil44.csv', index=False)
    stores.to_csv('./data/store_44/stores44.csv', index=False)
    transactions.to_csv('./data/store_44/transactions44.csv', index=False)

    print('Pre-processing complete.  Cleaned data sets saved to: ./data/store_44/')
