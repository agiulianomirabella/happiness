import pandas as pd

def read_data(year=None):
    if year:
        return __read_year(year)
    df = __read_year(2015)
    for year in [2016, 2017, 2018, 2019]:
        df = pd.concat([df, __read_year(year)], axis=0)
    df['country'] = df['country'].astype('category')
    return df

def __read_year(year):
    df = pd.read_csv('../data/'+ str(year) +'.csv')
    df['year'] = year
    return df