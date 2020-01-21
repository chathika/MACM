import pandas as pd
import gzip
import json
import numpy as np
import datetime as dt
import os, sys
import re
import glob
from pprint import pprint
from multiprocessing import Pool

#_folder_name = '/home/social-sim/MACMWorking/MACM/DryRunCp3_Sc2/Inputs/Exo'
#_nodelistfile = '/home/social-sim/MACMWorking/MACM/DryRunCp3_Sc2/Inputs/cp3_dry_run_s2_nodelist.txt'

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def GenerateTimeSeriesFromGDELT(folder_name):
    #output_file_name = 'GDELT_WhiteHelmets_Syria.csv'
    #output_directory = '/home/social-sim/MACMWorking/MACM/DryRunCp3_Sc2/all_exogenous_data/WhiteHelmets'
    nodelistfile = os.path.join(folder_name,'cp3_dry_run_s2_nodelist.txt')
    TS = []
    nodelist = []
    with open(nodelistfile) as fnl:
        nodelist = fnl.readlines()

    for i in range(len(nodelist)):
        nodelist[i] = nodelist[i][:-1]

    indexedKeywords = []
    keywords_to_nodes = {}
    for nd in nodelist:
        if nd[:8] == 'https://':
            keywords_to_nodes[nd] = [nd]
            indexedKeywords.append(nd)
        else:
            for k in nd.replace('_','.').split('-'):
                if k in keywords_to_nodes.keys():
                    keywords_to_nodes[k].append(nd)
                else:
                    keywords_to_nodes[k] = [nd]
                    indexedKeywords.append(k)

    pattern = ''
    i = 0
    for kw in indexedKeywords:
        pattern = pattern + '(?P<kw' + str(i) + '>' + kw + ')|'
        i += 1
    pattern = pattern[:-1]
    prog = re.compile(pattern)

    #---
    for gdeltQueryFile in glob.glob(folder_name + '/wh_gdelt_*json.gz'):
        print('File : ' + gdeltQueryFile)
        with gzip.GzipFile(gdeltQueryFile, 'r') as fin:
            multi_json_bytes = fin.read()

        multi_json_str = multi_json_bytes.decode('utf-8')
        json_str_list = multi_json_str.split('\n')
        print('starting...')
        for i in range(len(json_str_list) - 1):
            data = json.loads(json_str_list[i])
            update_progress(float(i) / float(len(json_str_list)))
            for v in data.values():
                for imatch in prog.finditer(str(v)):
                    if not imatch.lastgroup is None:
                        kid = int(imatch.lastgroup[2:])
                        for nid in keywords_to_nodes[ indexedKeywords[kid] ]:
                            TS.append([data['day'],nid,data['GoldsteinScale']])
    #---

    df = pd.DataFrame(TS, columns=['time','actor','gs'])
    print("1")
    df['time'] = df.apply(lambda x: dt.datetime.strptime(x.time,"%Y-%m-%dT%H:%M:%S"),axis=1)
    df = df.sort_values(by='time')
    return pd.pivot_table(df, columns=['actor'], index='time', values='gs',aggfunc=np.mean).resample("H").mean().ffill().fillna(0)