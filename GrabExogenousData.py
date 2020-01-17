
from datetime import datetime 
import traceback
import json
import datetime
from dateutil import parser
from datetime import datetime
import pandas as pd
import os
import re
from scipy.signal import freqs
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters

from pandas.io.json import json_normalize
import json
import glob
import argparse
import math

import GenerateTimeSeriesFromNVD

register_matplotlib_converters()

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', help='Specify the directory of the exogenous data folders',
                type=str)
ap.add_argument('-c', '--challenge', help='Specify the challenge to choose which types of data to process',
                type=int)
ap.add_argument('-s', '--scenario', help='Specify the scenario to choose which types of data to process',
                type=int)
ap.add_argument('-o', '--outputdir', help='Specify the directory to write the output files',
                type=str)
args = ap.parse_args()

if args.directory:
    DIRECTORY = args.directory
else:
    print("specify directory")
    exit()

if args.challenge == 2:
    if args.scenario:
        if args.scenario == 1:
            DATA_FOLDERS = ["hackernews","nvd"]
        elif args.scenario == 2:
            DATA_FOLDERS = ["Crypto_Price"]
    else:
        print("specify scenario")
        exit()
elif args.challenge == 3:
    if args.scenario:
        if args.scenario == 1:
            DATA_FOLDERS = ["NVD"]
        elif args.scenario == 2:
            DATA_FOLDERS = ["WhiteHelmets"]
    else:
        print("specify scenario")
        exit()

if args.outputdir:
    OUTDIR = args.outputdir
else:
    print("specify output directory")
    exit()

def detect_outlier_position_by_fft(signal, threshold_freq=0.5, frequency_amplitude=0.1):
    fft_of_signal = np.fft.fft(signal)
    fftfreq_of_signal = np.fft.fftfreq(len(signal))
    outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
    
    fft_AboveFreqThreshold = fft_of_signal[np.where(np.abs(fftfreq_of_signal) >= threshold_freq)]
    fftfreq_AboveFreqThreshold = fftfreq_of_signal[np.where(np.abs(fftfreq_of_signal) >= threshold_freq)]
    if len(fft_of_signal) == 0 or len(fftfreq_of_signal) == 0:
        return None
    fft_AboveThresholds = fft_AboveFreqThreshold[np.where(np.abs(fft_AboveFreqThreshold) > frequency_amplitude)]
    fftfreq_AboveThresholds = fftfreq_AboveFreqThreshold[np.where(np.abs(fft_AboveFreqThreshold) > frequency_amplitude)]
    if np.any(fftfreq_AboveThresholds):
        index_of_outlier = np.where(signal == outlier)
        return index_of_outlier[0][0]
    else:
        return None

def getOutliers(crypto_data, name, field, delta, freq_amp_threshold=0.1):    
    fig, (ax1) = plt.subplots(1,1)
    fig.set_size_inches(18.5, 10.5)
    ax1.scatter(crypto_data.index, crypto_data, c= "#348ABD", marker = ".", label = "Original Signal")
    ax1.set_title(name + " " + field +  " Weekly Difference Outliers")
    outlier_positions = []
    crypto_data_norm = crypto_data/ crypto_data.max()
    for ii in range(2* delta, crypto_data_norm.size, 1):
        outlier_position = detect_outlier_position_by_fft(crypto_data_norm.tolist()[ii-delta:ii+delta],frequency_amplitude=freq_amp_threshold)
        if outlier_position is not None:
            outlier_positions.append(ii + outlier_position - delta)
    outlier_positions = list(set(outlier_positions))
    outliers = crypto_data.iloc[outlier_positions]
    outlier_times = np.array(crypto_data.index.tolist())[outlier_positions]
    try:
        ax1.scatter(outlier_times, outliers, c= "#E24A33", marker = ".", label = "Outliers")
        ax1.set_xlim([datetime(2017,1,1),datetime(2018,4,1)])
        abs_lim = max(abs(crypto_data.min()),abs(crypto_data.max()))
        adjustment = abs(crypto_data.max() - crypto_data.min()) * 0.1
        ax1.set_ylim([-abs_lim - adjustment, abs_lim + adjustment])
        plt.savefig(os.path.join(OUTDIR,name +"_" + field + "_Difference_with_Outliers.png"))
    except:
        print("Warning plot not generated for " + name + "_" + field)
        pass
    outlier_df = pd.DataFrame(data = {"timestamp": crypto_data.index})
    outlier_df["outlier"] = outlier_df["timestamp"].isin(outlier_times.tolist())
    outlier_df = outlier_df.set_index("timestamp")
    
    return outlier_df


def extractOutliers():
    all_exogenous_data = {}
    data_folders = DATA_FOLDERS
    for data_folder in data_folders:
        print("datafolder " + str(data_folder))
        if 0 < len(glob.glob(os.path.join(DIRECTORY,data_folder) + '/nvdcve-1.0*.json.gz')):
            df = GenerateTimeSeriesFromNVD.GenerateTimeSeriesFromNVD(os.path.join(DIRECTORY,data_folder),50)
            print(df)
            all_exogenous_data[data_folder] = df
        else:
            for f in os.listdir(os.path.join(DIRECTORY, data_folder)):
                print("f " + str(f))
                print("Reading file: " + str(os.path.join(DIRECTORY,data_folder,f)))
                if re.search(".csv", f):
                    df = pd.read_csv(os.path.join(DIRECTORY,data_folder,f),parse_dates=True)
                    datetime_column = ""
                    for column in df.columns:
                        if "time" == column.lower() or "month" == column.lower() or "year" == column.lower() or "modified" == column.lower() or "date" == column.lower():
                            datetime_column = column
                            print(datetime_column)
                    try:
                        df[datetime_column] = df[datetime_column].astype('datetime64[s]')
                    except:
                        try:
                            df[datetime_column] = pd.to_datetime(df[datetime_column], format="%h/%m/%s", errors='coerce')
                        except:
                            print(os.path.join(DIRECTORY,data_folder,f))
                            print(df[datetime_column].head())
                            traceback.print_exc()
                            pass
                    pass
                    if datetime_column == "":
                        print(df.columns)
                    df = df.sort_values(by=datetime_column)
                    df = df.set_index(datetime_column)
                    all_exogenous_data[data_folder + "_" + f] = df
                if re.search(".json", f) or re.search("HNI", f):
                    try:
                        with open(os.path.join(DIRECTORY,data_folder,f)) as data_file:    
                            df= pd.DataFrame(json.load(data_file))
                        #df = json_normalize(d)
                        all_exogenous_data[data_folder + "_" + f] = df
                    except Exception:
                        with open(os.path.join(DIRECTORY,data_folder,f)) as data_file:    
                            if "major_cyber_incidents" in f.lower():
                                new_file = "["
                                for line in data_file:
                                    if "}" in line:
                                        new_file = new_file + line + ","
                                    else:
                                        new_file = new_file + line
                                new_file = new_file[:-1] + "]"
                                #with open (os.path.join("ExogenousData/",data_folder,"new"),"w") as fout:
                                    #fout.write(new_file)
                                df= pd.DataFrame(json.loads(new_file))
                            else:
                                df= pd.read_json(os.path.join(DIRECTORY,data_folder,f),lines=True)
                    try:
                        datetime_column = ""
                        for column in df.columns:
                            if "time" in column.lower() or "month" in column.lower() or "year" in column.lower() or "modified" in column.lower() or "date" in column.lower():
                                datetime_column = column
                        if "HNI" in f:
                            df[datetime_column] = df[datetime_column].astype('datetime64[s]')
                            #print(df)
                        else:
                            df[datetime_column] = df[datetime_column].astype('datetime64[ns]')
                        
                    except:
                        try:
                            df[datetime_column] = pd.to_datetime(df[datetime_column], format="%B %Y")
                            
                        except:
                            print("Error parsing dates for " + os.path.join(DIRECTORY,data_folder,f))
                            print(df[datetime_column].head())
                            traceback.print_exc()
                            pass
                    pass
                    df = df.sort_values(by=datetime_column)
                    df = df.set_index(datetime_column)
                    all_exogenous_data[data_folder + "_" + f] = df
            
    crypto_exogenous_shocks = pd.DataFrame()
    nvd_exogenous_shocks = pd.DataFrame()
    whitehelmets_exogenous_shocks = pd.DataFrame()
    # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    all_hackernews_data = pd.DataFrame()
    print(all_exogenous_data.keys())
    for name in all_exogenous_data.keys() :    
        print("Extracting exogenous shocks from: " + str(name))
        if "Crypto_Price" in name:
            name = name[len("Crypto_Price")+1:-4]
            crypto_data = all_exogenous_data[name]  
            crypto_volumeto = crypto_data["volumeto"].diff()      
            outlier_volumeto_df = getOutliers(crypto_volumeto,name,"volumeto", 3)
            outlier_volumeto_df.columns = [name[13:-5] + "_" + "volumeto"]

            crypto_open = crypto_data["open"].diff()      
            outlier_open_df = getOutliers(crypto_open,name,"open", 3)
            outlier_open_df.columns = [name[13:-5] + "_" + "open"]

            if not crypto_exogenous_shocks.shape[0] == 0:
                crypto_exogenous_shocks = crypto_exogenous_shocks.join(outlier_open_df, how="outer", lsuffix='_left', rsuffix='_right')
            else: 
                crypto_exogenous_shocks = outlier_open_df
            crypto_exogenous_shocks = crypto_exogenous_shocks.join(outlier_volumeto_df, how="outer", lsuffix='_left', rsuffix='_right')
            
        elif "hackernews" in name:
            hackernews_data = all_exogenous_data[name]
            if not all_hackernews_data.shape[0] == 0:
                all_hackernews_data = all_hackernews_data.append(hackernews_data)
            else:
                all_hackernews_data = hackernews_data
        elif "WhiteHelmets" in name:
            whitehelmet_data = all_exogenous_data[name]
            print("Starting " + str(name))
            for col in whitehelmet_data.columns:
                col_shocks = getOutliers(whitehelmet_data[col],name,col,3).rename(columns={"outlier":col})
                print("Completed for "+ str(col))
                whitehelmets_exogenous_shocks = whitehelmets_exogenous_shocks.join(col_shocks, how="outer", lsuffix='_left', rsuffix='_right')
        elif "NVD" in name:
            nvd_exogenous_data = all_exogenous_data[name]
            print(nvd_exogenous_data)
            print("Starting " + str(name))
            for col in nvd_exogenous_data.columns:
                col_shocks = getOutliers(nvd_exogenous_data[col],name,col,3).rename(columns={"outlier":col})
                print("Completed for "+ str(col))
                nvd_exogenous_shocks = nvd_exogenous_shocks.join(col_shocks, how="outer", lsuffix='_left', rsuffix='_right')

    all_hackernews_shocks=pd.DataFrame()
    if all_hackernews_data.shape[0]>0:
        all_hackernews_data = all_hackernews_data.sort_index()
        all_hackernews_data = all_hackernews_data[all_hackernews_data["type"] == "story"]
        all_hackernews_data_unique = all_hackernews_data.drop_duplicates(["id"])
        all_hackernews_data = all_hackernews_data.resample("3600S").max().reset_index().groupby(["time"]).mean()
        all_hackernews_data = all_hackernews_data / all_hackernews_data.max()
        all_hackernews_data = all_hackernews_data.diff()
        all_hackernews_shocks = getOutliers(all_hackernews_data.score,"hackernews","score",3,0.4)
        all_hackernews_shocks.columns = ["hackernews_shock"]
        print(all_hackernews_shocks)
    
    all_exogenous_shocks = pd.DataFrame()
    if crypto_exogenous_shocks.shape[0] != 0 and all_hackernews_shocks.shape[0] != 0 and nvd_exogenous_shocks.shape[0] != 0:
        all_exogenous_shocks = crypto_exogenous_shocks.join(all_hackernews_shocks, how="outer").join(nvd_exogenous_shocks, how="outer")
    elif crypto_exogenous_shocks.shape[0] == 0 and all_hackernews_shocks.shape[0] != 0 and nvd_exogenous_shocks.shape[0] != 0:
        all_exogenous_shocks = all_hackernews_shocks.join(nvd_exogenous_shocks, how="outer")
    elif crypto_exogenous_shocks.shape[0] != 0 and all_hackernews_shocks.shape[0] == 0 and nvd_exogenous_shocks.shape[0] == 0:
        all_exogenous_shocks = crypto_exogenous_shocks
    elif nvd_exogenous_shocks.shape[0] != 0:                
        all_exogenous_shocks = nvd_exogenous_shocks
    elif whitehelmets_exogenous_shocks.shape[0] != 0:
        all_exogenous_shocks = whitehelmets_exogenous_shocks
    all_exogenous_shocks = all_exogenous_shocks.fillna(0).astype(int)
    all_exogenous_shocks = all_exogenous_shocks.groupby(all_exogenous_shocks.index.date).sum() 
    all_exogenous_shocks.index = pd.to_datetime(all_exogenous_shocks.index).strftime('%Y-%m-%d %H:%M:%S')
    OUTPUT_FILE = os.path.join(OUTDIR ,"all_exogenous_shocks_cp{}_scen{}.csv".format(args.challenge,args.scenario))
    all_exogenous_shocks.to_csv(OUTPUT_FILE)
    print("File Written : " + OUTPUT_FILE)
    # all_exogenous_shocks.replace(True,1).replace(False,0).fillna(0).to_csv(OUTDIR + "all_exogenous_shocks_scen{}.csv".format(args.scenario), index=False)


    #plt.show()

def readOutliersAndSubsetForSimulation(_startDate,_stopDate):

    file_name = DIRECTORY + "/all_exogenous_shocks.csv"
    exogenous_shocks = pd.read_csv(file_name, parse_dates=["timestamp"]).sort_values("timestamp")
    exogenous_shocks = exogenous_shocks[(exogenous_shocks["timestamp"] > _startDate) & (exogenous_shocks["timestamp"] < _stopDate)].set_index("timestamp").resample("D").sum()
    exogenous_shocks = exogenous_shocks.drop(exogenous_shocks.columns[exogenous_shocks.apply(lambda col: col.sum() == 0)], axis = 1)
    print(exogenous_shocks.head())
    exogenous_shocks.to_csv(OUTDIR +"/Simulation_Ready_Exogenous_Shocks.csv", index=True, date_format = "%Y-%m-%dT%H:%M:%SZ")



extractOutliers()

# _startDate = datetime.strptime("2016-01-01T00:00:00Z","%Y-%m-%dT%H:%M:%SZ")
# _stopDate = datetime.strptime("2020-08-31T00:00:00Z","%Y-%m-%dT%H:%M:%SZ")
# readOutliersAndSubsetForSimulation(_startDate,_stopDate)

