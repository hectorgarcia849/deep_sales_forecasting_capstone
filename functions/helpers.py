# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:43:49 2019

@author: -
"""

from sklearn.metrics import mean_squared_error
import pandas as pd

from scipy.stats import norm
from scipy.stats import zscore
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import boxcox
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import yeojohnson


def get_nan_values_report(dict_df):
        res = {}
        for key in dict_df.keys():
            n = len(dict_df[key]['data'])
            cols = dict_df[key]['data'].columns
            for c in cols:
                nas = dict_df[key]['data'][c].isna().sum()
                res[key] = {c: { 'na_count': nas, 'na%': nas/n }}
        return res
    
def get_descriptive_stats(dict_df):
    res = {}
    
    for key in dict_df.keys():
        cols = dict_df[key]['data'].columns
        for c in cols:
            stats = dict_df[key]['data'].describe()
            res[key] = {'stats': stats} 
    return res

def standardize(series):
    mean = series.mean()
    std = series.std() 
    return (series-mean)/std

def std_item_analysis(data):
    items = data.item_nbr.unique()
    items_volume = pd.Series([])
    items_std = pd.Series([])
    res = {'highest_std(Q95)': None, 
           'lowest_std(Q5)': None, 
           'median_std': None, 
           'mean_std': None}

    for item in items:
        x = data.loc[data['item_nbr'] == item, 'unit_sales']
        volume = x.sum()
        std = x.std()        
        items_volume[item] = volume
        items_std[item] = std
        
    highest_std_q95_idx = abs(items_std - items_std.quantile(.95)).idxmin()
    lowest_std_q95_idx = abs(items_std - items_std.quantile(.05)).idxmin()

    res['highest_std(Q95)'] = (highest_std_q95_idx, items_std.loc[highest_std_q95_idx])
    res['lowest_std(Q5)'] = (lowest_std_q95_idx, items_std.loc[lowest_std_q95_idx])
    
    std_median_idx = items_std.sort_values().index[int(len(items_std)/2)]
    mean_std_idx = abs(items_std.sort_values() - items_std.mean()-1).idxmin()
            
    res['median_std'] = (std_median_idx, items_std.loc[std_median_idx])
    res['mean_std'] = (mean_std_idx, items_std.loc[mean_std_idx])
    print('length of mean_std: ', len(data.loc[data['item_nbr'] == mean_std_idx, 'unit_sales']))
    
    return res

def stationarity_analysis(X, item_analysis_res):
    res = pd.DataFrame([], columns=['kpss(unit_sales)','kpss(boxcox(unit_sales))', 'kpss(unit_sales.diff(1))', 'kpss(unit_sales.diff(365))', 'kpss(unit_sales.diff(1).diff(365))', 'normal_test(unit_sales)', 'shapiro_test(unit_sales)', 'normal_test(boxcox(unit_sales))', 'shapiro_test(boxcox(unit sales))', 'normal_test(boxcox(resid))', 'shapiro_test(boxcox(resid))'])
    
    for k in item_analysis_res.keys():
        item, _ = item_analysis_res[k]
        
        x = X.loc[X['item_nbr'] == item, ['date','unit_sales']]
        
        x.index = x['date']
        x = x.drop(['date'], axis=1)
        
        x_bc, x_bc_params = boxcox(x['unit_sales'])
        
        daily = 365
        decompose_x = seasonal_decompose(x['unit_sales'], model='additive', freq=daily)
        
        alpha = 0.05
        kpss_test = 'stationary' if kpss(x['unit_sales'], regression='ct')[1] > alpha else '-'
        kpss_test_bc = 'stationary' if kpss(boxcox(x['unit_sales'])[0], regression='ct')[1] > alpha else '-'
        kpss_test_diff1 = 'stationary' if kpss(x['unit_sales'].diff(1), regression='ct')[1] > alpha else '-'
        kpss_test_diff365 = 'stationary' if kpss(x['unit_sales'].diff(365), regression='ct')[1] > alpha else '-'
        kpss_test_diff1_365 = 'stationary' if kpss(x['unit_sales'].diff(365), regression='ct')[1] > alpha else '-'        
        norm_test = 'normal' if normaltest(x['unit_sales'])[1] >= alpha else '-'
        shapiro_test = 'normal' if shapiro(x['unit_sales'])[1] >= alpha else '-'
        norm_test_bc = 'normal' if normaltest(boxcox(x['unit_sales'])[0])[1] >= alpha else '-'
        shapiro_test_bc = 'normal' if shapiro(boxcox(x['unit_sales'])[0])[1] >= alpha else '-'
        norm_test_resid_bc = 'normal' if normaltest(yeojohnson(decompose_x.resid.dropna())[0])[1] >= alpha else '-'
        shapiro_test_resid_bc = 'normal' if shapiro(yeojohnson(decompose_x.resid.dropna())[0])[1] >= alpha else '-'
        
        
        res.loc[k, :] =  (kpss_test, kpss_test_bc, kpss_test_diff1, kpss_test_diff365, kpss_test_diff1_365, norm_test, shapiro_test, norm_test_bc, shapiro_test_bc, norm_test_resid_bc, shapiro_test_resid_bc)
    
    return res

def corr_analysis(X, item_analysis_res, confidence=.95):
    res = pd.DataFrame([], columns=['sig_auto_lags', 'sig_partial_lags'])
    for k in item_analysis_res.keys():
        item, _ = item_analysis_res[k]
        x = X.loc[X['item_nbr'] == item, 'unit_sales']
        auto = acf(x)
        partial = pacf(x)
        z_auto = zscore(auto)
        z_partial = zscore(partial)
        cdf_auto = norm.cdf(z_auto)
        cdf_partial = norm.cdf(z_partial)
        print()
        ss_auto_lags = [i[0] for i in filter(lambda x:x[1][2] >= confidence, enumerate(zip(auto, z_auto, cdf_auto)))]
        ss_partial_lags = [i[0] for i in filter(lambda x:x[1][2] >= confidence, enumerate(zip(partial, z_partial, cdf_partial)))]
        
        res.loc[k, :] = (ss_auto_lags, ss_partial_lags)
    
    return res
    


#def autogenerate_arima(data):
#    """ Inspired by proceess outlined in Hyndman et al. """
#
#    # stablize variance -- use box-cox
#    
#    # test for stationarity / difference
#
#    items = data['item_nbr'].unique().tolist()
#    stationary = {}
#
#    for i in items:
#        
#        """ Decompose then perform tests"""
#        
#        stationary[i] = {}
#        
#        try:
#            train_p_value = kpss(data.loc[data['item_nbr'] == i, 'unit_sales'])[1]
#        except:
#            train_p_value = 0
#        
#        try:
#            first_p_value = kpss(data.loc[data['item_nbr'] == i, 'unit_sales'].diff().dropna(0))[1]
#        except:
#            first_p_value = 0
#            
#        try:
#            seas_p_value = kpss(data.loc[data['item_nbr'] == i, 'unit_sales'].diff(365).dropna())[1]
#        except:
#            seas_p_value = 0
#        
#        try:
#            first_seas_p_value = kpss(data.loc[data['item_nbr'] == i, 'unit_sales'].diff().diff(365).dropna())[1]
#        except:
#            first_seas_p_value = 0
#        
#        stationary[i]['train'] = train_p_value == .1
#        stationary[i]['first_diff'] = first_p_value == .1
#        stationary[i]['seas_diff'] = seas_p_value == .1
#        stationary[i]['first_seas_diff'] = first_seas_p_value == .1
#        
#        # ARIMA(0,d,0)
#        # ARIMA(2, d, 2)
        
    
        