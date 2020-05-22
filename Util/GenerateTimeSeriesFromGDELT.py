import pandas as pd
import gzip
import json
import numpy as np
import datetime as dt
import os, sys
import re
import glob
from pprint import pprint
import multiprocessing

prog = 0
indexedKeywords = []
keywords_to_nodes = {}
MainDF = 0

def processData(startpos, endpos):
        if endpos - startpos < 1:
            return 0
        partialTS = []
        for idx,row in MainDF.iloc[list(np.arange(startpos,endpos))].iterrows():
            for jdx,v in row.iteritems():
                for imatch in prog.finditer(str(v)):
                    if not imatch.lastgroup is None:
                        kid = int(imatch.lastgroup[2:]) #get id number
                        for nid in keywords_to_nodes[ indexedKeywords[kid] ]:
                            partialTS.append([row['day'],nid,row['GoldsteinScale']])
        return partialTS

def GenerateTimeSeriesFromGDELT(folder_name):
    global prog
    global indexedKeywords
    global keywords_to_nodes
    global MainDF
    
    nodelistfile = os.path.join(folder_name,'cp3_s2_nodelist.txt')
    nodelist = []
    with open(nodelistfile) as fnl:
        nodelist = fnl.readlines()
    
    for i in range(len(nodelist)):
        nodelist[i] = nodelist[i][:-1]
    
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

    TS = []

    #---
    for gdeltQueryFile in glob.glob(folder_name + '/wh_gdelt_q*json.gz'):
        print('\t\tcurrently collected rows ' + str(len(TS)))
        print('\t\tReading File : ' + gdeltQueryFile)
        MainDF = pd.read_json(gdeltQueryFile,compression='gzip',lines=True)
        print('\t\tRows to Proces ' + str(MainDF.shape[0]))
        NProcs = multiprocessing.cpu_count()
        print('\t\tNum of Procs ' + str(NProcs))
        sizePerProc = 10000
        print('\t\tRows per Proc ' + str(sizePerProc))
        rowsPerRun = sizePerProc * NProcs
        print('\t\tRows per Run ' + str(rowsPerRun))
        currentRowStart = 0
        currentRowEnd = rowsPerRun
        while currentRowEnd < MainDF.shape[0]:
            print('\t\tProcessing from ' + str(currentRowStart) + ' to ' + str(currentRowEnd))
            paramList = [(currentRowStart + i * sizePerProc, currentRowStart + (i + 1) * sizePerProc ) for i in range(NProcs - 1) ]
            if currentRowEnd > paramList[-1][1]:
                paramList.append((paramList[-1][1], currentRowEnd))
            print('\t\tAllocation:')
            s = ''
            for p in paramList:
                s += '[' + str(p[0]) + '-' + str(p[1]) + '),'
            print(s)
            with multiprocessing.Pool(NProcs) as p:
                results = p.starmap(processData, paramList)
            for pr in results:
                TS = TS + pr
            print('\t\tcurrently collected rows ' + str(len(TS)))
            currentRowStart = currentRowEnd
            currentRowEnd = currentRowStart + rowsPerRun
        
        print('\t\tdone looping the file.')
        if currentRowStart < MainDF.shape[0] and currentRowStart < currentRowEnd :
            thegap = MainDF.shape[0] - currentRowStart
            print('\t\tremaining rows ' + str(thegap) )
            numProcsReq = thegap // sizePerProc
            print('\t\tRequired Procs ' + str(numProcsReq))
            paramList = [(currentRowStart + i * sizePerProc, currentRowStart + (i+1) * sizePerProc) for i in range(numProcsReq - 1)]
            print(paramList)
            if paramList == []:
                paramList = [(currentRowStart, MainDF.shape[0])]
            if MainDF.shape[0] > paramList[-1][1]:
                paramList.append((paramList[-1][1], MainDF.shape[0]))
            print(paramList)
            s = ''
            for p in paramList:
                s += '[' + str(p[0]) + '-' + str(p[1]) + '),'
            print(s)
            with multiprocessing.Pool(NProcs) as p:
                results = p.starmap(processData, paramList)
            for pr in results:
                TS = TS + pr
        
        print('\t\tcurrently collected rows ' + str(len(TS)))
            
    #---
    print('\t\tFinally collected rows ' + str(len(TS)))
    MainDF = pd.DataFrame(TS, columns=['time','actor','gs'])
    #MainDF.to_csv('/home/social-sim/MACMWorking/MACM/DryRunCp3_Sc2/Inputs/ALLExoGrabOutput/test.csv')
    print("GDELT DataFrame Generation Done.")
    MainDF['time'] = MainDF.apply(lambda x: dt.datetime.strptime(x.time,"%Y-%m-%dT%H:%M:%S"),axis=1)
    MainDF = MainDF.sort_values(by='time')
    return pd.pivot_table(MainDF, columns=['actor'], index='time', values='gs',aggfunc=np.mean).resample("H").mean().ffill().fillna(0)