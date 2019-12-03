# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:52:25 2019

@author: -
"""
""" https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/ """


# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


class SarimaxSearch:
    
    def __init__(self, data, n_test=4):
        self.data = data
        self.n_test = n_test
        self.config_space = SarimaxSearch.sarima_configs()
        
    # one-step sarima forecast
    @staticmethod
    def sarima_forecast(history, config):
        order, sorder, trend = config
    	# define model
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    	# fit model
        model_fit = model.fit(disp=False)
    	# make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0], model_fit
     
    # root mean squared error or rmse
    @staticmethod
    def measure_rmse(actual, predicted):
    	return sqrt(mean_squared_error(actual, predicted))
     
    # split a univariate dataset into train/test sets
    @staticmethod
    def train_test_split(data, n_test):
    	return data[:-n_test], data[-n_test:]
     
    @staticmethod
    def walk_forward_validation(data, n_test, config):
        
        predictions = list()
        train, test = SarimaxSearch.train_test_split(data, n_test)
        history = [x for x in train]
        model = None
        
        for i in range(len(test)):
    		# fit model and make forecast for history
            yhat, model = SarimaxSearch.sarima_forecast(history, config)
    		# store forecast in list of predictions
            predictions.append(yhat)
    		# add actual observation to history for the next loop
            history.append(test[i])
    	# estimate prediction error
        error = SarimaxSearch.measure_rmse(test, predictions)
        return error, model
     
    # score a model, return None on failure
    @staticmethod
    def score_model(data, n_test, config, debug=False):
        score = None
        model = None
        key = str(config)
    
        if debug:
            score, model = SarimaxSearch.walk_forward_validation(data, n_test, config)
            
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    score, model = SarimaxSearch.walk_forward_validation(data, n_test, config)
            except:
                score = None
        if score is not None:
            print(' > Model[%s] %.3f' % (key, score))
    
        return (key, score, model)
     
    # grid search configs
    def grid_search(self, parallel=True):
        scores = None
        if parallel:
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(SarimaxSearch.score_model)(self.data, self.n_test, config) for config in self.config_space)
            scores = executor(tasks)
        else:
            scores = [self.score_model(self.data, self.n_tests, config) for config in self.config_space]
        scores = [r for r in scores if r[1] != None]
        scores.sort(key=lambda tup: tup[1])
        return scores
     
    # create a set of sarima configs to try
    @staticmethod
    def sarima_configs(seasonal=[0]):
    	params = list()
    	# define config lists
    	p_params = [0, 1, 2]
    	d_params = [0, 1]
    	q_params = [0, 1, 2]
    	t_params = ['n','c','t','ct']
    	P_params = [0, 1, 2]
    	D_params = [0, 1]
    	Q_params = [0, 1, 2]
    	m_params = seasonal
    	# create config instances
    	for p in p_params:
    		for d in d_params:
    			for q in q_params:
    				for t in t_params:
    					for P in P_params:
    						for D in D_params:
    							for Q in Q_params:
    								for m in m_params:
    									config = [(p,d,q), (P,D,Q,m), t]
    									params.append(config)
    	return params
     
            
