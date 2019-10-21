import pandas as pd
import numpy as np
import seaborn as sns

def present_section(section_name):
    print ("\n###" + section_name + "###\n")    
    
def present_option_for_next_section(msg):
    input("\nPress Enter to " + msg +"...\n")



print("...............................................................................")
print("Exploring and Cleaning the Data")
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

""" Version 1, looking at unit sales data, there are some outlier of instances where unit sales hit very levels.
    The data set comes with the information that Ecaudor suffered an Earthquake on April 16, 2016 and people rallied to
    buy water and other first aid needs -- this is apparent in the data set.  The median of unit sales is 4.0, the mean is 8.55
    the max unit sales is 8944.0.
"""

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
    100 of these items where on promo that day.  Nan under holiday, means no holiday.
    
    
          date  gt_200_count   holiday  gt_200_promo
    2017-06-04           306      NaN         100.0                          

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

""" Items data set provides details of individual items sold, many of the columns are categorical and converted into 
    dummy variables.
"""
items['perishable'] = items['perishable'].astype('uint8')
items = pd.concat([items, pd.get_dummies(items['family'])], axis=1).drop(['family'], axis=1)
items = pd.concat([items, pd.get_dummies(items['class'])], axis=1).drop(['class'], axis=1)


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

"""
    Transactions
"""
transactions['date'] = pd.to_datetime(transactions['date'], infer_datetime_format=True)
transactions = pd.concat([transactions, pd.get_dummies(transactions['store_nbr'])], axis=1).drop(['store_nbr'], axis=1)
transactions['transactions'] = transactions['transactions'].astype('uint16')

print("Join the relevant tables to create training data set and test set")

""" Joins: train and test into one dataset (for now). NOTE stopped here for now, due to memory error issues when
    trying to join datasets. """
train_n = len(train.index)
test_n = len(test.index)
test.index = pd.RangeIndex(train_n, train_n + test_n)


print("...............................................................................")
print("Exploratory Data Analysis")
print("...............................................................................\n")





