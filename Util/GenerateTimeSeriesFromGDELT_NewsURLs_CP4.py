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
import math
from flashtext.keyword import KeywordProcessor

prog = 0
indexedKeywords = [] # keywords are the urls
keywords_to_narratives = {}
MainDF = 0

def createRegex(iStart, indexedKeywordsSlice):
    pattern = ''
    i = iStart
    for kw in indexedKeywordsSlice:
        pattern = '{}(?P<kw{}>{})|'.format(pattern, str(i), re.escape(kw))
        i += 1
    return pattern


def processData(startpos, endpos):
        if endpos - startpos < 1:
            return 0
        partialTS = []
        print('\tstarted:{} to {}'.format(startpos,endpos))
        for idx,row in MainDF.iloc[list(np.arange(startpos,endpos))].iterrows():
            for v in ( row['sourceurl'], row['sourceurl_h'] ):
                for kwmatch in prog.extract_keywords(str(v)):
                    for nid in keywords_to_narratives[kwmatch]:
                        partialTS.append([row['day'],nid,row['GoldsteinScale']])
                # for imatch in prog.finditer(str(v)):
                #     if not imatch.lastgroup is None:
                #         kid = int(imatch.lastgroup[2:]) #get id number
                #         for nid in keywords_to_narratives[ indexedKeywords[kid] ]:
                #             partialTS.append([row['day'],nid,row['GoldsteinScale']])
        print('\t\tended:{} to {}'.format(startpos,endpos))
        return partialTS

def GenerateTimeSeriesFromGDELT_NewsURLs_CP4(folder_name):
    global prog
    global indexedKeywords
    global keywords_to_narratives
    global MainDF

    dfArticleNar = pd.read_csv(os.path.join(folder_name,'labeledArticlesWithURLs.csv'))
    indexedUrlList = list(dfArticleNar.url.unique())

    for i, row in dfArticleNar.iterrows():
        keywords_to_narratives[row['url']] = [ row['narrative'] ]
        indexedKeywords.append(row['url'])
    
    # Create regex and compile it for text match
    # print('Creating the regex...')
    # NProcs = multiprocessing.cpu_count()
    # SliceSize = int(math.floor(len(indexedKeywords) / NProcs))
    # paramList = [ (i * SliceSize, indexedKeywords[i * SliceSize : (i + 1) * SliceSize]) for i in range(NProcs - 1) ]
    # paramList.append(( (NProcs - 1) * SliceSize, indexedKeywords[(NProcs - 1) * SliceSize :] ))
    # with multiprocessing.Pool(NProcs) as p:
    #     results = p.starmap(createRegex, paramList)
    # pattern = ''.join(results)[:-1]
    # print('Compiling the regex...')
    # prog = re.compile(pattern)

    #keyword processor
    prog = KeywordProcessor()
    for kw in indexedKeywords:
        prog.add_keyword(kw)

    TS = []

    #---
    for gdeltQueryFile in glob.glob(folder_name + '/*gdelt*json.gz'):
        print('\t\tcurrently collected rows ' + str(len(TS)))
        print('\t\tReading File : ' + gdeltQueryFile)
        MainDF = pd.read_json(gdeltQueryFile,compression='gzip',lines=True)
        MainDF = MainDF.drop(columns=['QuadClass', 'MonthYear', 'ActionGeo_ADM1Code', 'EventBaseCode',
                                        'EventCode', 'NumSources',
                                        'Actor1Geo_CountryCode', 'EventRootCode',
                                        'ActionGeo_Long', 'Actor1Geo_ADM2Code',
                                        'Actor1Code', 'ActionGeo_CountryCode', 
                                        'Actor1CountryCode', 'Actor1Geo_FeatureID', 'Actor2Geo_Long',
                                        'IsRootEvent', 'Actor2Geo_CountryCode', 'Actor1Geo_Type',
                                        'Actor2Geo_FeatureID', 'globaleventid', 'ActionGeo_Type',
                                        'Actor2Geo_ADM1Code', 'Actor1Geo_Long', 'Actor2Geo_ADM2Code',
                                        'Actor1Geo_ADM1Code', 'Actor2Code', 'dateadded', 'Actor2CountryCode',
                                        'Actor2Geo_Type', 'ActionGeo_FeatureID', 'filename',
                                        'FractionDate', 'Year', 'NumMentions', 'Actor2Geo_Lat', 'Actor1Geo_Lat',
                                        'ActionGeo_ADM2Code', 'ActionGeo_Lat', 'NumArticles',
                                        'Actor1Type1Code', 'Actor2Type1Code', 'Actor1Type2Code',
                                        'Actor2KnownGroupCode', 'Actor2Type2Code', 'Actor1KnownGroupCode',
                                        'Actor1Religion2Code', 'Actor1Religion1Code', 'Actor2Type3Code',
                                        'Actor2Religion2Code', 'Actor2Religion1Code', 'Actor2EthnicCode',
                                        'Actor1EthnicCode', 'Actor1Type3Code',
                                        'Actor1Name', 'Actor2Name', 'ActionGeo_FullName', 'Actor1Geo_FullName', 'Actor2Geo_FullName'
                                        ])
        print(MainDF)
        print(MainDF.columns)
        print('\t\tRows to Proces ' + str(MainDF.shape[0]))
        NProcs = multiprocessing.cpu_count() - 1
        print('\t\tNum of Procs ' + str(NProcs))
        sizePerProc = 1000
        print('\t\tRows per Proc ' + str(sizePerProc))
        rowsPerRun = sizePerProc * NProcs
        print('\t\tRows per Run ' + str(rowsPerRun))
        currentRowStart = 0
        currentRowEnd = rowsPerRun
        while currentRowEnd < MainDF.shape[0]:
            print('\t\tProcessing from ' + str(currentRowStart) + ' to ' + str(currentRowEnd))
            paramList = [(currentRowStart + i * sizePerProc, currentRowStart + (i + 1) * sizePerProc) for i in range(NProcs - 1) ]
            if currentRowEnd > paramList[-1][1]:
                paramList.append((paramList[-1][1], currentRowEnd ))
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
            paramList = [(currentRowStart + i * sizePerProc, currentRowStart + (i+1) * sizePerProc ) for i in range(numProcsReq - 1)]
            #print(paramList)
            if paramList == []:
                paramList = [(currentRowStart, MainDF.shape[0])]
            if MainDF.shape[0] > paramList[-1][1]:
                paramList.append((paramList[-1][1], MainDF.shape[0]))
            #print(paramList)
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
    print("GDELT DataFrame Generation Done.")
    MainDF['time'] = MainDF.apply(lambda x: dt.datetime.strptime(x.time,"%Y-%m-%dT%H:%M:%S"),axis=1)
    MainDF = MainDF.sort_values(by='time')
    MainDF.to_csv(os.path.join(folder_name,'MACMProcessedGDELT.csv'))
    return pd.pivot_table(MainDF, columns=['actor'], index='time', values='gs',aggfunc=np.mean).resample("H").mean().ffill().fillna(0)