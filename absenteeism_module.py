#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


#  custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.std_ = np.std(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_scaled, X_not_scaled], axis=1)[init_col_order]

    
# prediction class
class absenteeism_model():
    
    def __init__(self, model, scaler):      
        # load saved model and scaler
        with open('absenteeism_model','rb') as model, open('custom_scaler','rb') as scaler:
            self.reg = pickle.load(model)
            self.scaler = pickle.load(scaler)
            self.data = None
            
    # load and preprocess data
    def load_and_clean_data(self, data):
        # load
        df = pd.read_csv(data, delimiter=',')
        # create palceholder
        self.df_with_predictions = df.copy()
        # drop ID
        df = df.drop(['ID'], axis=1)
        # to enable the use of previous code
        df['Absenteeism Time in Hours'] = 'Nan'
        
        # create Reason dummies
        reason_dummies = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        # reason cols
        reason_type_1 = reason_dummies.loc[:,1:14].max(axis=1)
        reason_type_2 = reason_dummies.loc[:,15:17].max(axis=1)
        reason_type_3 = reason_dummies.loc[:,18:21].max(axis=1)
        reason_type_4 = reason_dummies.loc[:,22:].max(axis=1)
        
        # drop Reason for Absence
        df = df.drop(['Reason for Absence'], axis=1)
        
        # concatenate onto df
        df = pd.concat([df,reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
        
        # rename and reorder columns
        column_names = ['Date', 'Transportation Expense',
               'Distance to Work', 'Age', 'Daily Work Load Average',
               'Body Mass Index', 'Education', 'Children', 'Pets',
               'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

        df.columns = column_names

        cols_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
               'Distance to Work', 'Age', 'Daily Work Load Average',
               'Body Mass Index', 'Education', 'Children', 'Pets',
               'Absenteeism Time in Hours']

        df = df[cols_reordered]
        
        # convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        # extract month and weekday
        df['Month'] = df['Date'].apply(lambda x: x.month)
        df['Weekday'] = df['Date'].apply(lambda x: x.weekday())
        # drop date
        df.drop(['Date'], axis=1, inplace=True)
        
        # reorder cols
        cols_reordered_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month',
       'Weekday', 'Transportation Expense', 'Distance to Work',
       'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[cols_reordered_upd]
        
        # remap education categories
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # replace missing values
        df = df.fillna(value=0)
        
        # drop original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        
        # drop unnecessary features
        df = df.drop(['Month','Daily Work Load Average', 'Distance to Work'],axis=1)
        
        # create preprocessed data attribute
        self.preprocessed_data = df.copy()
        
        self.data = self.scaler.transform(df)
    
    # function for predicting probability of outcome as 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
        
    # function to output 0 or 1 based on model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    # predict outputs and probability and add to data as columns
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

