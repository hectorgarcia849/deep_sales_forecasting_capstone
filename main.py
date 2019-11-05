import pandas as pd
import numpy as np
import seaborn as sns
import math
import json
import pickle

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy.stats import boxcox
from scipy.stats import zscore
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from itertools import product


def present_section(section_name):
    print ("\n###" + section_name + "###\n")    
    
def present_option_for_next_section(msg):
    input("\nPress Enter to " + msg +"...\n")

print("...............................................................................")
print("Exploring and Cleaning the Data -- Version 1 only look at 2 stores")
print("...............................................................................\n")

train = pd.read_csv('./data/train.csv')
holidays = pd.read_csv('./data/holidays_events.csv')
items = pd.read_csv('./data/items.csv')
oil = pd.read_csv('./data/oil.csv')
stores = pd.read_csv('./data/stores.csv')
transactions = pd.read_csv('./data/transactions.csv')
test = pd.read_csv('./data/test.csv')

all_data = {\
        'train': {'data': train}, \
        'holidays': {'data': holidays}, \
        'items': {'data': items}, \
        'oil': {'data': oil}, \
        'stores': {'data': stores}, \
        'transactions': {'data': transactions}, \
        'test': {'data': test}}


print("Search for missing values (NaNs, blanks, nulls...) and impute for missing values")

# double check that missing values aren't due to holidays

def get_nan_values_report(dict_df):
    res = {}
    for key in dict_df.keys():
        n = len(dict_df[key]['data'])
        cols = dict_df[key]['data'].columns
        for c in cols:
            nas = dict_df[key]['data'][c].isna().sum()
            res[key] = {c: { 'na_count': nas, 'na%': nas/n }}
    return res

print("NaN Values summary:")
nan_values = get_nan_values_report(all_data)
print(nan_values)

""" Version 1, train data is given to us with 17% of rows having nan for onpromotion.  
    Assuming these are False.  In later version, will explore deeper for a better
    imputation strategy."""

train['onpromotion'] = train['onpromotion'].fillna(False)

""" Version 1, oil data is given with 3.5% of rows missing data in column 'dcoilwtico'
    This is the price of oil.  Missing data will be imputed by taking previous days oil price,
    except for the first row where there is no previous day -- in this case will use the next
    day's oil price.
"""

oil.loc[0, 'dcoilwtico'] = oil.loc[1, 'dcoilwtico']
oil = oil.fillna(method='ffill')
oil.index = oil['date']
oil = oil.drop(['date'], axis=1)
""" 
    To prepare for boxcox transformation data must remove all negatives
"""

train['unit_sales'].loc[train['unit_sales'] <= 0] = np.nan

def get_descriptive_stats(dict_df):
    res = {}
    
    for key in dict_df.keys():
        cols = dict_df[key]['data'].columns
        for c in cols:
            stats = dict_df[key]['data'].describe()
            res[key] = {'stats': stats} 
    return res

stats = get_descriptive_stats(all_data)



print(train['unit_sales'].describe())


""" Digging deeper into the skew of unit sales, we can see that only 0.114% of days is there unit sales that exceed 200 sales
    on an item at store.  April 18 records the highest number of hits, 456, while April 19 record 369 hits.  Near the top is also
    the days around Christmas, which is no surprise due to the expected seasonal demand increase.
    
    Other notable exceptions, April 1, 2017 and June 4, 2017.  Require further investigation On April 2 2017 Ecuador had a national
    election, which may be associated with this unusual increase in purchases.  Reuters reports there was a risk
    of similar type of gov't that could take power as in Venezula.  
    https://www.reuters.com/article/us-ecuador-election-venezuela/venezuela-crisis-casts-shadow-over-ecuador-presidential-election-idUSKBN1733U2
    
    June 4, 2017 -- Unclear why there was such a surge of sales on this day.  Otherwise, the list looks as one expected, holidays coinciding with
    increased sales. See joined list between days with highest sales and holidays.  
    
    Promotions explain some of the dates with higher number of sales that don't occur on holidays -- a count of promos that saw sales
    greater than 200 is joined with the table as well.  For example, June 4, 2017 has 306 instances where items sell over 200, we know that
    100 of these items where on promo that day.  
    
    Nan under holiday, means no holiday.
    
    
          date  gt_200_count   holiday  gt_200_promo
    2017-06-04           306      NaN         100.0                 

    Strategy will be to include categorical variable for within x days of Earthquake to mark the jump in sales, in other cases where
    outside of 95% confidence interval -- will make NaN then use backfill strategy to impute.       

"""



n_train = len(train)
n_unit_sales_gt_200 = pd.DataFrame(train[train['unit_sales'] > 200]['date'].value_counts())
n_unit_sales_gt_200 = n_unit_sales_gt_200.reset_index(level=0)
n_unit_sales_gt_200.columns = ['date', 'gt_200_count']
print()

cross_ref_holiday = pd.merge(n_unit_sales_gt_200, holidays[['date', 'description']], on=['date', 'date'], how='left')

promos_gt_200 = train.loc[(train['onpromotion'] == 1) & (train['unit_sales'] > 200)]['date'].value_counts()
promos_gt_200 = promos_gt_200.reset_index(level=0)
promos_gt_200.columns = ['date', 'gt_200_promo']

cross_ref_promo = pd.merge(cross_ref_holiday, promos_gt_200, on=['date', 'date'], how='left')

print(cross_ref_promo.loc[0:24])


print("Check and convert data types to appropriate types")

""" Train data set, shows date and on promotion as objects, in pandas this could mean they are strings.  Same conversions apply to test set.

    1) Will convert date to a date type, which will allow for computations on dates such as searching within ranges of a date.  Wi
    2) Will convert date to a int type, 0 for false and 1 for true, as learning algorithms will require numbers.
"""
print(train.dtypes)

train['onpromotion'] = train['onpromotion'].astype('uint8')
train['date'] = pd.to_datetime(train['date'], infer_datetime_format=True)

test['onpromotion'] = test['onpromotion'].astype('uint8')
test['date'] = pd.to_datetime(test['date'], infer_datetime_format=True)

""" Since unit sales are not very large numbers, i.e. """


""" Holidays data set, requires converting column 'date' from string to a date type object 
    
    column 'type' from string object to dummies
    column 'locale' from string object to 
    column 'locale_name' from string object to 
    column 'description' from string to
    column 'transferred' from bool to int8
    
    Note about transferred column -- A holiday that is transferred officially falls on that 
    calendar day, but was moved to another date by the government. 
    A transferred day is more like a normal day than a holiday. 
    To find the day that it was actually celebrated, look for the corresponding row where type is Transfer
"""
holidays['date'] = pd.to_datetime(holidays['date'], infer_datetime_format=True)
holidays = pd.concat([holidays, pd.get_dummies(holidays['type'])], axis=1).drop(['type'], axis=1)
holidays = pd.concat([holidays, pd.get_dummies(holidays['locale'])], axis=1).drop(['locale'], axis=1)
holidays = pd.concat([holidays, pd.get_dummies(holidays['locale_name'])], axis=1).drop(['locale_name'], axis=1)
holidays = pd.concat([holidays, pd.get_dummies(holidays['description'])], axis=1).drop(['description'], axis=1)
holidays['transferred'] = holidays['transferred'].astype('uint8')
holidays.index = holidays['date']
holidays = holidays.drop(['date'], axis=1)


""" Items data set provides details of individual items sold, many of the columns are categorical and converted into 
    dummy variables.
"""
items['perishable'] = items['perishable'].astype('uint8')
items = pd.concat([items, pd.get_dummies(items['family'])], axis=1).drop(['family'], axis=1)
items = pd.concat([items, pd.get_dummies(items['class'])], axis=1).drop(['class'], axis=1)
items.index = items['item_nbr']
items = items.drop(['item_nbr'], axis=1)


"""
    Stores provides details about individuals stores including what cluster they are part of (group of similar stores)
    In this data set:
        Converting categorical variables into dummies: city, state, type
        Converting cluster and store_nbr from int64 to uint8 (to reduce memory usage)
        
"""

stores = pd.concat([stores, pd.get_dummies(stores['city'])], axis=1).drop(['city'], axis=1)
stores = pd.concat([stores, pd.get_dummies(stores['state'])], axis=1).drop(['state'], axis=1)
stores = pd.concat([stores, pd.get_dummies(stores['type'])], axis=1).drop(['type'], axis=1)
stores['store_nbr'] = stores['store_nbr'].astype('uint8')
stores['cluster'] = stores['cluster'].astype('uint8')
stores.index = stores['store_nbr']
stores = stores.drop(['store_nbr'], axis=1)

"""
    Transactions
"""
transactions['date'] = pd.to_datetime(transactions['date'], infer_datetime_format=True)
transactions['transactions'] = transactions['transactions'].astype('uint16')

""" 
    Outliers
    Version 1, looking at unit sales data, there are some outlier of instances where unit sales hit very levels.
    The data set comes with the information that Ecaudor suffered an Earthquake on April 16, 2016 and people rallied to
    buy water and other first aid needs -- this is apparent in the data set.  The median of unit sales is 4.0, the mean is 8.55
    the max unit sales is 8944.0.
    
    With ouliers, we may wish to replace them with missing values, or with an estimate that is more consistent 
    with the majority of the data; depending on the context.  However, this will need to be done by item at the store
    level.
    
    Possible future improvements to below -- take holidays into account, do it by store and item grouped
    
    
    item_nbrs = items['item_nbr'].values.tolist()
    store_nbrs = stores['store_nbr'].values.tolist()
    
    count = 1
    for i in product(store_nbrs, item_nbrs):
        count += 1
        print(count)
        s = train[['unit_sales', 'date', 'store_nbr', 'item_nbr']].loc[train['store_nbr'] == i[0]]
        i = s.loc[train['item_nbr'] == i[1]]
        i['zscore'] = zscore(i['unit_sales']) 
        idx = i.loc[(abs(i['zscore']) >= 3)].index
        train['unit_sales'].loc[idx] = np.nan

"""
sales_mean = train['unit_sales'].mean()
sales_std = train['unit_sales'].std()
z_score = lambda x: ((x - sales_mean) / sales_std) >= 3

train['unit_sales_outlier'] = z_score(train['unit_sales'])


""" Initial clean up, at this point realized that data is too big for -- moving to Spark """

#train.to_csv('./clean/train.csv')
#holidays.to_csv('./clean/holidays_events.csv')
#items.to_csv('./clean/items.csv')
#oil.to_csv('./clean/oil.csv')
#stores.to_csv('./clean/stores.csv')
#transactions.to_csv('./clean/transactions.csv')
#test.to_csv('./clean/test.csv')




print("...............................................................................")
print("Exploratory Data Analysis")
print("...............................................................................\n")

""" 
    Time Series Plots grouped by store, items
"""

by_store = train[['date', 'store_nbr', 'unit_sales']].groupby(['store_nbr', 'date']).sum()
by_store.unstack(level=0).plot(legend=False)

by_item = train[['date', 'item_nbr', 'unit_sales']].groupby(['item_nbr', 'date']).sum()
by_item.unstack(level=0).plot(legend=False)


"""
    Explore Seasonality & Trend by store, by cluster, by item
"""

daily = 365
decompose_by_store = seasonal_decompose(by_store.unstack().T.fillna(0), model='additive', freq=daily)
decompose_by_store.plot()

""" If adfuller null hypothesis rejected, then data has not unit root and is stationary (mean / variance consistent over time)  """

# First Difference, Seasonal Difference, First Difference + Seasonal Difference   
""" It is important that if differencing is used, the differences are interpretable. 
    First differences are the change between one observation and the next. Seasonal 
    differences are the change between one year to the next."""
    
# First Difference
first_diff_by_store = by_store.unstack().T[54].diff().fillna(0)
first_diff_by_store.hist()
# Seasonal Difference
seasonal_diff_by_store = by_store.unstack().T[54].diff(365).fillna(0)
seasonal_diff_by_store.hist()

# KPSS null hypothesis is that the data is stationary, adfuller null hypothesis is that it is not stationary

# ARIMA(p,d,q): p - number of lagged obs, d - number of times raw obs are differenced, q - size of the moving average window
# choosing p - plot autocorrelation see how long autocorrelations remain positive / negative and are statistically sign

# Autocorrelation checks for autocorrelation at a variety of k lags -- however y, yt-1, yt-2 are correlated
# because of their connection to yt-1.  Partial autocorrelations allow you to measure y and yt-2 without assuming
# autocorrelation with yt-1 i.e. with the relationships of intervening observations removed.  Both are interesting plots to look at.  

# statistical significant correlation if autocorrelation value lies outside of the 95% confidence interval
plot_acf(by_item.unstack().T.fillna(0)[103501].loc[('unit_sales')], lags=54)
plot_pacf(by_item.unstack().T.fillna(0)[103501].loc[('unit_sales')], lags=54)
pyplot.show() 

# use AIC and BIC together in model selection. For example, in selecting the number of latent classes in a model, 
# if BIC points to a three-class model and AIC points to a five-class model, it makes sense to select from models 
# with 3, 4 and 5 latent classes. AIC is better in situations when a false negative finding would be considered 
# more misleading than a false positive, and BIC is better in situations where a false positive is as misleading 
# as, or more misleading than, a false negative.

# note AIC, AICc and BIC can be used to select p, q parameters but not d, different evaluation measure requried for d.


# AR model
# I model
# q model
# ARIMA model

""" Model Building """

""" 
    ARIMA on store 44 (max volume store), by item, by family, by class.  Strategy here is to train
    an ARIMA model for each item listed in Store 44, save those models to disk and load them later into
    a dictionary.  The dictionary object that stores the models is later used for predictions, using
    each model for predictions.
"""

""" By item """

X_44 = train.loc[train['store_nbr'] == 44].reset_index().drop(['index', 'id'], axis=1)
assortment_44 = X_44['item_nbr'].unique()

X_44_by_item = {}

for i in assortment_44:
    X_44_by_item[i] = X_44.loc[X_44['item_nbr'] == i].sort_values(['date'])[['date', 'unit_sales']]
    X_44_by_item[i].index = X_44_by_item[i]['date']
    X_44_by_item[i] = X_44_by_item[i]['unit_sales']

models = {}

for i in assortment_44:
    a_pdq = [[0], 0, 0]
    s_pdq = [[0], 0, 0]
    models[i] = {}
                
    """ First difference & check if data is stationary """
    X = X_44_by_item[i].fillna(method='ffill')
    X_diff = X_44_by_item[i].diff().fillna(0)
    is_stationary = adfuller(X_diff)[1] <= 0.05
    models[i]['data'] = { 'is_stationary': is_stationary }
    a_pdq[1] = 1
    
    """ On differenced data, find auto-correlations and select p for ARIMA
        
    autocorr = acf(X, qstat=True, nlags=50, alpha=0.05)
    plot_acf(X, lags=1000, alpha=0.05) -- based on visual inspection will search an in n lags up to 7.
    
    """
    a_pdq[0] = 5
    
    """ Start with 0 for q"""
    a_pdq[2] = 0  
    
    
    """ Split into train and test """
    train_44_size = int(len(X) * 0.80)
    train_44, test_44 = X[0:train_44_size].tolist(), X[train_44_size: len(X)].tolist()
        
    history = [x for x in train_44]
    predictions = []
    model = None
    
    for t in range(len(test_44)):
        model = ARIMA(history, order=(tuple(a_pdq)))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_44[t]
        history.append(obs)
#            print('predicted=%f, expected=%f' % (yhat, obs))
    summary = { 'ARIMA' + str((p, a_pdq[1], a_pdq[2])): {
            'MSE': mean_squared_error(test_44, predictions), \
            'MAE': mean_absolute_error(test_44, predictions), \
            'RMSE': math.sqrt(mean_squared_error(test_44, predictions))}}
    
    with open('./summary/arima/store#{}_item#{}.json'.format(44,i), 'w') as outfile:
        json.dump(summary, outfile)
    
    with open('./models/arima/store#{}_item#{}.pkl'.format(44,i), 'wb') as outfile:
        pickle.dump(model_fit, outfile)
    
    print('complete item number {}'.format(i))
        
path = './models/arima'
store_44_models = os.listdir(path)
store_44_dict = {}

for m in store_44_models:
    print(m)
    with open(path+'/'+m, 'rb') as file:
        store_44_dict[m] = pickle.load(file)

print('dict of models for store 44 complete!')
        
#        models[i]['ARIMA' + str((p, a_pdq[1], a_pdq[2]))] = \
#        {'MSE': mean_squared_error(test_44, predictions), \
#         'MAE': mean_absolute_error(test_44, predictions), \
#         'RMSE': math.sqrt(mean_squared_error(test_44, predictions)), \
#         'model': model_fit}
#        


# Building bench marks from traditional methods
# naive
# seasaonal naive
# Exponential Smoothing
# ARIMA
# SARIMA
# ARIMAX
# SARIMAX 

# RNN
# CNN

model = ARIMA(by_item.unstack().T.fillna(0)[103501].loc[('unit_sales')].values, order=(10,1,5))


model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# take seasonal difference first, if still non-stationary then take a first difference -- plot this and test this.
# however you decide to difference, this is then entered as the d, D values
# based on this differenced data, review auto-correlations to select p and q 
# 
smodel = SARIMAX(by_item.unstack().T.fillna(0)[103501].loc[('unit_sales')].values,\
                 order=(6,1,3), \
                 seasonal_order=(3,1,3,365), \
                 enforce_stationarity=False, enforce_invertibility=False)

smodel_fit = smodel.fit(disp=0)
print(smodel_fit.summary())
smodel_fit.plot_diagnostics(figsize=(15, 12))


# can use sklearn for evaluation metrics
pred = sarima.predict(tr_end,te_end)[1:]
print('SARIMA model MSE:{}'.format(mean_squared_error(tes,pred)))

pd.DataFrame({'test':test,'pred':pred}).plot()
plt.show()

# another check for how good the model (S)ARIMA is to ensure you are minimizing auto-correlations in the residual
# Can use the The Ljung-Box test to check for autocorrelations in the residual.  When you include exogenous variables
# you will have two error terms -- one for regression error term (n) where a test called Breusch-Godfrey test for assessing whether 
# the resulting residuals were significantly correlated.  The regression residuals an have autocorrelation though.
# For the ARIMA model error term (e) we can test for auto-correlations using The Ljung-Box test
# When using (S)ARIMAX only the ARIMA erorr is assumed to be white noise; the regression error is not assumed so, it
# can have autocorrelations in it.  In these combined models -- minimizing e not n is done.

# when including exogenous variables, all data must be made stationary -- this can be achieved with a first difference
# where the data is not stationary.  If all the variables are stationary, then only need to consider ARMA errors for
# the residuals.

# For data with long seasonal periods USE: STL, Dynamic Harmonic Regression,  TBATS -
# - such as daily data with annual seasonality use haromonic regression approach
# where the seasonal pattern is modelled using Fourier terms with short-term dynamics handled by ARMA error.
# Advantages -- allows for any length seasonality, for more than one seasonal period Fourier terms of different frequencies
# can be included, smoothness of the seasonal pattern can be controlled by K the number of fourier sin and cos pairs 
# (smoother for smaller values of K)
# Disadvantage that sesonality is fixed and not allowed to change over time -- use less data and re-train.


# ETS models are designed to handle this situation by allowing the trend and seasonal terms 
# to evolve over time. ARIMA models with differencing have a similar property. But dynamic 
# regression models do not allow any evolution of model components.

residuals = pd.DataFrame(smodel_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
