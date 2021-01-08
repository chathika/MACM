import pandas as pd
import datetime as dt
import glob
import os
import numpy as np

def GenerateTimeSeries_GDELT_CP5(folder_name):
    file_name = glob.glob(os.path.join(folder_name , '*cpec.exogenous.gdelt.events.v1.labeled*.json'))[0]
    df = pd.read_json(file_name,lines=True)
    df = df[df.supervised_frames.apply(lambda x: len(x) > 0)][['day','supervised_frames','GoldsteinScale']]
    df['time'] = df.day.apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S"))
    df['impact'] = df.GoldsteinScale
    df['label'] = df.supervised_frames
    df = df.drop(columns=['day','GoldsteinScale','supervised_frames'])
    rows = []
    _ = df.apply(lambda r: [rows.append([r.time, r.impact, lbl]) for lbl in r.label] , axis=1)
    df = pd.DataFrame(rows, columns=['time','impact','label'])
    #df.to_csv('/home/social-sim/Desktop/CJ_MACM_Deploy/macm_deploy/MACM/init_data/test.csv',index=False)
    return pd.pivot_table(df, columns=['label'], index='time', values='impact',aggfunc=np.mean).resample('H').mean().ffill().fillna(0)