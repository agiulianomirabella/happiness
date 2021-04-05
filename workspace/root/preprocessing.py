from root.parameters import TARGET
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

def one_hot_encoding(df):
    return pd.get_dummies(df, prefix_sep='==', dummy_na=True)
    # return pd.get_dummies(df, prefix_sep='==', columns= enums(df), dummy_na=True)

def numeric_encoding(df):
    years = df.pop('year')
    columns = [c for c in df.columns if not df[c].isnull().values.all()]
    df = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df), columns=columns)
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    df['year'] = years
    return df

def txt2enum(df):
    for column in texts(df):
        df[column] = df[column].apply(lambda x: x if pd.isna(x) else 'DUMMY')
        df[column] = df[column].astype('category')
    return df

def regressor_preprocess(df):
    df = df.drop('country', axis=1)
    df = txt2enum(df)
    df = one_hot_encoding(df)
    df = numeric_encoding(df)
    return df

def numericals(df):
    out = list(df.select_dtypes(include = 'number').columns)
    if TARGET in out:
        out.remove(TARGET)
    return out

def texts(df):
    out = list(df.select_dtypes(include = 'object').columns)
    if TARGET in out:
        out.remove(TARGET)
    return out

def enums(df):
    out = list(df.select_dtypes(include = ['category', 'bool']).columns)
    if TARGET in out:
        out.remove(TARGET)
    return out

