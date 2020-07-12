import pandas as pd
import numpy as np
from datetime import datetime as dt
import glob
import os

def GenerateTimeSeriesFromNewsArticles(folder_name):
    file_name = glob.glob(os.path.join(folder_name , '*timeseries*.csv'))[0]
    df = pd.read_csv(file_name,parse_dates=['date'])
    df = df[df.date != 'xyz']
    df['time'] = df.date
    #df['time'] = df.date.apply(lambda x: dt.strptime(x,'%Y-%m-%d '))
    df['impact'] = df['count']
    return pd.pivot_table(df, columns=['label'], index='time', values='impact',aggfunc=np.mean).resample('H').mean().ffill().fillna(0)