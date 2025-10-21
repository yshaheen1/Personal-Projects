# imports
from IPython.display import display, HTML
display(HTML("<style>.container { width:99% !important; }</style>"))

import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import json
import matplotlib
import statsmodels.api as sm
import itertools

from datetime import datetime

# functions

def lag_transfo(series, lag=1):
    series_len = len(series)
    return_series = np.concatenate((series.values[0:1].repeat(lag), series.values[:series_len - lag]))
    return return_series


def adstock_transfo(series, adstock = 0.3):
    return_series = series + lag_transfo(series, 1)*(adstock) + lag_transfo(series, 2)*(adstock**2) + lag_transfo(series, 3)*(adstock**3)
    return return_series

def coeff_transfo(series_X, feature, target_series, coeff):  # Freeze the coeffs
    return_target_series = target_series - coeff * series_X[feature]  
    series_X = series_X.drop([feature], axis = 1)    # remove feature from the features that will be fitted
    return series_X, return_target_series

def run_model(db, features_set, target_column):
    coeff_set = {}  # set to return parameters
    model_y = db[target_column]
    features_list = list(features_set.keys())
    model_X = db[features_list]
    for feature in features_list:
        if 'lag' in features_set[feature]:
            model_X[feature] = lag_transfo(model_X[feature], features_set[feature]['lag'])
        if 'adstock' in features_set[feature]:
            model_X[feature] = adstock_transfo(model_X[feature], features_set[feature]['adstock'])
        if 'coeff' in features_set[feature]:    
            model_X, model_y = coeff_transfo(model_X, feature, model_y, features_set[feature]['coeff'])
            coeff_set[feature] = features_set[feature]['coeff']


    model = sm.OLS(model_y, model_X)
    model2 = model.fit()

    # create series series with the coeffs
    coeff_series = model2.params
    new_coeffs = pd.Series(coeff_set) 
    coeff_series = pd.concat([coeff_series, new_coeffs])

    return model2, coeff_series


# Dealing with lists of adstock and lag

def model_r2(db, features_set, target_column, lag_dic, adstock_dic, coeff_dic):  # lag_dic and adstock_dic should be dictionaries with only numbers as values (no list)
    new_db = db.copy()
    model_y = db[target_column]
    features_list = list(features_set.keys())
    model_X = db[features_list]
    for feature, lag in lag_dic.items():
        model_X.loc[:,feature] = lag_transfo(model_X[feature], lag)
        new_db[feature] = model_X[feature]
    for feature,adstock in adstock_dic.items():
        model_X.loc[:,feature] = adstock_transfo(model_X[feature], adstock)
        new_db[feature] = model_X[feature]
    for feature, coeff in coeff_dic.items():
        model_X, model_y = coeff_transfo(model_X, feature, model_y, coeff)
    
    model = sm.OLS(model_y, model_X)
    model2 = model.fit()

    return model2.rsquared, model2, new_db

# get a list of adstock and lag possibilities
def possibilities(no_list_dic, list_dic):
    return_list = []
    list_possibilities = better_cartesian_product(list_dic)
    for possib in list_possibilities:
        dic = no_list_dic.copy()
        dic.update(possib)
        return_list.append(dic)
    return return_list

# compute cartesian product, useful for possibilities function
def cartesian_product(dic_of_lists, list_of_dics):
    list_keys = list(dic_of_lists.keys())
    i = 0
    while (i < len(list_keys) - 1) & (type(dic_of_lists[list_keys[i]]) != type([])):
        i += 1
    if (i < len(list_keys) - 1) or (type(dic_of_lists[list_keys[i]]) == type([])):
        for j in dic_of_lists[list_keys[i]]:
            new_dic_of_lists = dic_of_lists.copy()
            new_dic_of_lists[list_keys[i]] = j
            cartesian_product(new_dic_of_lists,list_of_dics)
    else:
        list_of_dics.append(dic_of_lists)

def better_cartesian_product(dic_of_lists):
    list_cart_features = list(dic_of_lists.keys())
    list_to_be_cart = []
    for feature in list_cart_features:
        list_to_be_cart.append(dic_of_lists[feature])
    list_cart = itertools.product(*list_to_be_cart)
    # create list containing treated dics
    list_of_dics = []
    len_dic =  len(list_cart_features)
    for tup in list_cart:
        new_dic = {}
        for i, feature in enumerate(list_cart_features):
            new_dic[feature] = tup[i]
        list_of_dics.append(new_dic)
    return list_of_dics

        

def run_best_model(db, features_set, target_column):
    # Create dictionaries with coefficients, and lag and adstock lists
    coeff_dic = {}
    lag_list_dic, adstock_list_dic = {}, {}
    lag_no_list_dic, adstock_no_list_dic = {},{}
    features_list = list(features_set.keys())
    for feature in features_list:
        if 'coeff' in features_set[feature]:
            coeff_dic[feature] = features_set[feature]['coeff']
        if 'lag' in features_set[feature]:
            if type(features_set[feature]['lag']) == type([]):
                lag_list_dic[feature] = features_set[feature]['lag']
            else:
                lag_no_list_dic[feature] = features_set[feature]['lag']
        if 'adstock' in features_set[feature]:
            if type(features_set[feature]['adstock']) == type([]):
                adstock_list_dic[feature] = features_set[feature]['adstock']
            else:
                adstock_no_list_dic[feature] = features_set[feature]['adstock']

    # iterate through lag and adstock possibilities and return the best rsquared and model
    best_lags, best_adstocks = {}, {}
    best_model = sm.OLS(db[target_column],db[features_list]).fit()   
    best_r2 = -10
    best_db = db.copy()
    lag_possibilities, adstock_possibilities = possibilities(lag_no_list_dic, lag_list_dic), possibilities(adstock_no_list_dic, adstock_list_dic)
    for lag_possibility in lag_possibilities:
        for adstock_possibility in adstock_possibilities:
            r2, mod, new_db = model_r2(db, features_set, target_column, lag_possibility, adstock_possibility, coeff_dic)
            if r2 > best_r2:
                best_r2, best_model, best_db = r2, mod, new_db
                best_adstocks, best_lags = adstock_possibility, lag_possibility
    
    coeff_series = best_model.params
    new_coeffs = pd.Series(coeff_dic) 
    calc_coeff_df = pd.concat([coeff_series, new_coeffs])
    return best_model, best_lags, best_adstocks, coeff_dic, calc_coeff_df, best_db


def timesplit_feat(db, feature, split_week, after=True):
    if after:
        direction = 'after'
    else:
        direction = 'before'
    new_feature_name = feature + '_' + direction + '_' + split_week
    return_db = db.copy()
    if after:
        return_db[new_feature_name] = return_db[feature].where(return_db['date'] >= split_week, other=0)
    else:
        return_db[new_feature_name] = return_db[feature].where(return_db['date'] < split_week, other=0)
    return return_db

def timewindow_feat(db, feature, start_week, end_week):

    new_feature_name = feature + '_' + start_week + '_' + end_week
    return_db = db.copy()
    return_db[new_feature_name] = return_db[feature].where((return_db['date'] >= start_week) & (return_db['date'] <= end_week), other=0)
    
    return return_db

def trend_feat(db, start_week, end_week,ascending=True):

    new_feature_name = 'trend' + '_' + start_week + '_' + end_week

    return_db = db.copy()

    trend_db = return_db[(return_db['date'] >= start_week) & (return_db['date'] <= end_week)][['date']].reset_index(drop=True).drop_duplicates()

    trend_length = len(trend_db['date'])

    trend = list(range(1,trend_length + 1))

    if ascending == False:
        trend.reverse()

    trend_db[new_feature_name] = trend

    return_db = return_db.merge(right=trend_db, how='left', on='date').fillna(0)

    return return_db

def window_feat(db, start_end_week_pairs, feat_name):
    
    return_db = db.copy()

    return_db[feat_name] = 0
    for pair in start_end_week_pairs:
        start_week = pair[0]
        end_week =pair[1]
        #new_feature_name = new_feature_name + '_' + start_wk + '_' + end_wk
        return_db[feat_name] = return_db[feat_name].mask((return_db['date'] >= start_week) & (return_db['date'] <= end_week), other=1)

    return return_db

def transform_db(database):
    db = database
    db = db.rename(columns={'week_starting_date':'date'})

    db['date'] = pd.to_datetime(db['date'])

    # DB summary data
    date_ct = db['date'].nunique()
    date_min = db['date'].min()
    date_max = db['date'].max()
    print(' db shape:', db.shape, '\n count of unique dates:',date_ct,'\n min date:', date_min,'\n max date:', date_max)

    ## load dep_mean into prism indexing format
    ## UPDATE THESE
    target_column = 'sales'
    date_column = 'date'
    
    return db


def get_features_list(database):
    db = database
    list_features = list(db.columns)
    new_list_features = []
    for feature in list_features:
        new_list_features.append("'"+feature + "':{'lag':[0,1],'adstock':[0.1,0.6]},")
    new_list_features
    i = 0
    for feature in list_features:
        i += 1
        if i<40:
            continue
        # print("'"+feature + "':{'lag':[0,1],'adstock':[0.1,0.6]},")
        print("'"+feature + "':{},")
    return new_list_features

def model_output(ols_model_coeffs, ols_adstocks, ols_lags, ols_model, features_set):
    # create DF to be pasted in Model sheet
    model_df = ols_model_coeffs.to_frame()
    model_df.columns = ['coeff']

    # Create list with features for which we fixed the coeff
    list_notnull_coeff = []
    for key in features_set.keys():
        if 'coeff' in features_set[key]:
            list_notnull_coeff.append(key)

    # Create adstock and lag lists
    adstock_list, lag_list = [],[] 
    for feature in model_df.index:
        if feature in list(ols_adstocks.keys()):
            adstock_list.append(ols_adstocks[feature])
        else:
            adstock_list.append(0)
        if feature in list(ols_lags.keys()):
            lag_list.append(ols_lags[feature])
        else:
            lag_list.append(0)

    # add adstock and lag columns to model_df
    model_df['lag'] = lag_list
    model_df['adstock'] = adstock_list
    model_df['saturation'] = ''

    # add t-stats to model_df
    t_values = ols_model.tvalues.tolist()
    for feature in list_notnull_coeff:
        t_values.append(0)
    model_df['t stat'] = t_values

    # order columns
    col_list = ['lag','adstock','saturation','coeff','t stat']
    model_df = model_df.reindex(columns = col_list)

    # paste the output to the Model sheet of the Analysis Tool
    return model_df

def inputsheet(database, model_df):
    # create df to paste in InputSheet sheet
    input_df = database.copy()

    # Add column with "paste here" and DMAs
    input_df.insert(loc = 0, column = 'Paste Here',value = 'All')

    # create the 4 rows at the top, and put values in first 2 columns
    fill_list = ['saturation_S','saturation_K', 'adstock', 'lag']
    for i in range(4):
        input_df.loc[-i-1] = np.nan
        input_df.loc[-i-1,'date'] = fill_list[i]
    input_df.loc[-1, 'Paste Here'] = 'province_cat'
    input_df.sort_index(inplace = True)
    input_df.head()

    # Fill values  in first 3 rows 
    for index in model_df.index:
        input_df.loc[-4, index] = model_df.loc[index,'lag']
        input_df.loc[-3, index] = model_df.loc[index,'adstock']

    # Paste the output in the InputSheet sheet of the AT. Clear before pasting
    return input_df