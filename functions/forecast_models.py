"""
    Starting place of SarimaxSearch class was inspired from an article:
    Title: How to Grid Search SARIMA Hyperparameters for Time Series Forecasting
    Author: Jason Brownlee
    Website: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/        
    Publish Date: August 5, 2019
    Accessed: Nov 2019
"""
#from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



class SarimaxSearch:
    
    def __init__(self, data, params, n_test=.2, seasonal_period=[7, 21]):
        self.data = data
        self.n_test = n_test
        print(params, data)
        self.config_space = SarimaxSearch.sarima_configs(params['p'], params['d'], params['q'], ['ct'], params['P'], params['D'], params['Q'], seasonal_period)
        
    @staticmethod
    def sarima_forecast(history, config):
        order, seasonal_order, trend = config
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0], model_fit
     
    @staticmethod
    def measure_rmse(actual, predicted):
        """ Root mean squared error could be understood as the standard deviation of the unexplained variance """
        return sqrt(mean_squared_error(actual, predicted))
    
    @staticmethod
    def measure_mae(actual, predicted):
        return mean_absolute_error(actual, predicted)
     
    @staticmethod
    def train_test_split(data, n_test):
        n = int(n_test * len(data))
        return data[:-n], data[-n:]
     
    @staticmethod
    def walk_forward_validation(data, n_test, config, scoring='mae'):
        
        if scoring not in ['avg', 'rmse', 'mae']:
            raise ValueError('Invalid value for scoring parameter.  Must be one of [avg, rmse, mae]')
        
        predictions = list()
        train, test = SarimaxSearch.train_test_split(data, n_test)
        history = [x for x in train]
        model = None
        
        for i in range(len(test)):
            yhat, model = SarimaxSearch.sarima_forecast(history, config)
            predictions.append(yhat)
            history.append(test[i])
    	

        rmse = SarimaxSearch.measure_rmse(test, predictions)
        mae = SarimaxSearch.measure_mae(test, predictions)
        avg = (rmse + mae) / 2
            
        score = avg if scoring == 'avg' else rmse if scoring == 'rmse' else mae 
        return score, model
     
    @staticmethod
    def score_model(data, n_test, config, debug=False):
        score = None
        model = None
    
        if debug:
            score, model = SarimaxSearch.walk_forward_validation(data, n_test, config)
            
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    score, model = SarimaxSearch.walk_forward_validation(data, n_test, config)
            except:
                score = None
#        print('\t config: ', config, 'mae: ', score, 'model: ', model.summary())
        return (config, score, model, data)
     
    def grid_search(self, parallel=True):
        scores = None
        print('# of configs: ', len(self.config_space))
        if parallel:
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(SarimaxSearch.score_model)(self.data, self.n_test, config) for config in self.config_space)
            scores = executor(tasks)
        
        else:
            scores = [self.score_model(self.data, self.n_test, config) for config in self.config_space]
        
        scores = [r for r in scores if r[1] != None]
        scores.sort(key=lambda tup: tup[1])
        
        return scores
     
        
    @staticmethod
    def sarima_configs(p_params, d_params, q_params, t_params, P_params, D_params, Q_params, seasonal=[0]):
        params = list()
        p_params = p_params
        d_params = d_params
        q_params = q_params
        t_params = t_params
        P_params = P_params
        D_params = D_params
        Q_params = Q_params
        m_params = seasonal

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
    

""" 
    Starting place of LSTM class was inspired from an article:
    Title: Time series forecasting
    Author: TensorFlow
    Website: https://www.tensorflow.org/tutorials/structured_data/time_series
    Publish Date: Unknown
    Accessed: Nov 2019
"""

""" 
Yoshua Bengio, et al., Learning Long-Term Dependencies with Gradient Descent is Difficult, 1994.

The paper defines 3 basic requirements of a recurrent neural network:

That the system be able to store information for an arbitrary duration.
That the system be resistant to noise (i.e. fluctuations of the inputs that are random or irrelevant to predicting a correct output).
That the system parameters be trainable (in reasonable time).
"""

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(13)

class UnivariateLSTMSearch():
    
    def __init__(self, data, batch_size=[128, 256, 512, 1024], eval_interval=[100, 200, 500], val_interval=[50, 100, 200], epochs=[10, 15, 20], buffer_size=1000, n_test=.2, history_size=20, target_size=0):
        self.data = data
        train_split = int(len(data)*n_test)
        self.x_train, self.y_train = UnivariateLSTMSearch.__univariate_data(data, 0, train_split, history_size, target_size)
        self.x_val, self.y_val = UnivariateLSTMSearch.__univariate_data(data, train_split, None, history_size, target_size)
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.eval_interval = eval_interval
        self.val_interval = val_interval
        self.epochs = epochs
        
        self.results = {}
    
    
    def run(self):
        
        config_space = self.__get_config_space()
        
        for config in config_space:
            batch_size, eval_interval, val_interval, epochs = config
            
            train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            train = train.cache().shuffle(self.buffer_size).batch(batch_size).repeat()
    
            val = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
            val = val.batch(batch_size).repeat()
        
            simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=self.x_train.shape[-2:]), tf.keras.layers.Dense(1)])
    
            simple_lstm_model.compile(optimizer='adam', loss='mae')
            history = simple_lstm_model.fit(train, epochs=epochs,
                          steps_per_epoch=eval_interval,
                          validation_data=val, validation_steps=val_interval)
            
            print(history.history)
            self.results[config] = history.history
        
        return self.results
    
    @staticmethod
    def __univariate_data(data, start_index, end_index, history_size, target_size):
        """ history_size is the relevant past observations to take into account 
            target_size refers to how far in the future predictions are required
        """
        
        x = []
        y = []
        
        print('data:', data)
        
        start_index = start_index + history_size
        
        if end_index is None:
            end_index = len(data) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            x.append(np.reshape(data[indices], (history_size, 1)))
            y.append(data[i + target_size])
            
        return np.array(x), np.array(y)

    
#    def plot(plot_data, delta, title):
#        labels = ['History', 'True Future', 'Model Prediction']
#        marker = ['.-', 'rx', 'go']
#        steps = UnivariateLSTMSearch.__create_time_steps(plot_data[0].shape[0])
#        
#        if delta:
#            future = delta
#        else:
#            future = 0
#        
#        plt.title(title)
#        
#        for i, x in enumerate(plot_data):
#            if i:
#                plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
#            else:
#                plt.plot(steps, plot_data[i].flatten(), marker[i], label=labels[i])
#        plt.legend()
#        plt.xlim([steps[0], (future+5)*2])
#        plt.xlabel('Time-Step')
#        return plt
    
    @staticmethod
    def __create_time_steps(n):
        steps = []
        for i in range(-n, 0, 1):
            steps.append(i)
        return steps
    
    def baseline(history):
        return np.mean(history)

    def __get_config_space(self):
        params = list()
                
        for batch in self.batch_size:
            for eva_int in self.eval_interval:
                for val in self.val_interval:
                    for e in self.epochs:
                        config = (batch, eva_int, val, e)
                        params.append(config)
        return params

