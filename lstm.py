# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:09:08 2019

@author: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

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
    
    def __init__(self, data, batch_size=[128], eval_interval=[200], val_interval=[100], epochs=[20], buffer_size=1000, n_test=.2, history_size=21, target_size=0):
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
            model = simple_lstm_model.fit(train, epochs=epochs,
                          steps_per_epoch=eval_interval,
                          validation_data=val, validation_steps=val_interval)
            
            print(model.history)
            self.results[config] = model
        
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

    
    def plot(plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        steps = UnivariateLSTMSearch.__create_time_steps(plot_data[0].shape[0])
        
        if delta:
            future = delta
        else:
            future = 0
        
        plt.title(title)
        
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
            else:
                plt.plot(steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
        return plt
    
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
        
        
""" Setting seed to ensure reproducibility."""

if __name__ == '__main__':
    
    data = pd.read_csv('./data/store_44/X44.csv')
    
    params_dict = {}
    params_dict['highest_std(Q95)'] = {'item_nbr': 1463797 }
    params_dict['lowest_std(Q5)'] = {'item_nbr': 1230461}
    params_dict['median_std'] = {'item_nbr': 580929}
    params_dict['mean_std'] = {'item_nbr': 111397}
    
    results = {}
    
    for k in params_dict.keys():
        item = params_dict[k]['item_nbr']
        print("LSTM MODEL BEING TRAINED ON: ", item)
        d = data.loc[data['item_nbr'] == item, 'unit_sales'].dropna().values
#        print(len(d))
        d_train_mean = d.mean()
        d_train_std = d.std()
        d_std = (d-d_train_mean)/d_train_std
        lstm = UnivariateLSTMSearch(d_std)
        lstm.run()
        print(lstm.results)
        results[k] = lstm
    
    outputfile = open('results_lstm', 'ab')
    pickle.dump(results, outputfile)
    outputfile.close()
    
    
