import pandas as pd
import gzip
import json
import numpy as np
import datetime as dt
import re
import os
import glob


def GenerateTimeSeriesFromNVD(folder_name, numOfTS = 100):
    TS = []
    for file_name in glob.glob(folder_name + "/nvdcve-1.0*.json.gz"):
        with gzip.GzipFile(file_name, 'r') as fin:
            json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        for cve_item_index in range(len(data['CVE_Items'])):
            for vendor_index in range(len(data['CVE_Items'][cve_item_index]['cve']['affects']['vendor']['vendor_data'])):
                for product_index in range(len(data['CVE_Items'][cve_item_index]['cve']['affects']['vendor']['vendor_data'][vendor_index]['product']['product_data'])):
                    TS.append([data['CVE_Items'][cve_item_index]['lastModifiedDate'], data['CVE_Items'][cve_item_index]["cve"]["CVE_data_meta"]['ID'].lower(), data['CVE_Items'][cve_item_index]['impact']['baseMetricV2']['impactScore']])
    df = pd.DataFrame(TS, columns=['time','product','impact'])
    df['time'] = df.apply(lambda x: dt.datetime.strptime(x.time,"%Y-%m-%dT%H:%MZ"),axis=1)
    df = df.sort_values(by='time')
    dfimpact = df.groupby('product').apply(lambda x: x.impact.mean()).abs().reset_index(name='sev').sort_values('product')
    dfcounts = df.groupby('product').count().sort_values('product').reset_index()
    TS = []
    for i,row in dfimpact.iterrows():
        prodCount = dfcounts[ dfcounts['product'] == row['product'] ].iloc[0]['impact']
        TS.append([ row['product'], row['sev'] * prodCount])
    df_products = pd.DataFrame(TS, columns = ['product', 'sev_pop_index'])
    if df_products.shape[0] < 500 :
        numOfTS = df_products.shape[0]
    print('Number of Time Serieses ' + str(numOfTS))
    topProducts = list(df_products.sort_values('sev_pop_index').tail(numOfTS)['product'].unique())
    dffocus = df[ df['product'].isin(topProducts) ]
    dffocus = dffocus.sort_values(by='time')
    return pd.pivot_table(dffocus, columns=['product'], index='time', values='impact',aggfunc=np.mean).resample("H").mean().ffill().fillna(0)