"""MACM, The Multi-Action Cascade Model of Conversation
Copyright (C) 2019  Chathika Gunaratne
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

"""
                    This code was authored by Chathika Gunaratne
And is the implementation for the IC2S2 presentation at the University of Amsterdam, 2019
If you use this code or the MACM model in your work, please cite using the following bibtex:

@inproceedings{gunaratne2019,
  author =       {Gunaratne, Chathika and Senevirathna, Chathurani and Jayalath, Chathura and Baral, Nisha and Rand, William and Garibay, Ivan},
  title =        {A Multi-Action Cascade Model of Conversation

},
  booktitle =    {5th International Conference on Computational Social Science},
  year =         {2019},
  url =          {http://app.ic2s2.org/app/sessions/9kXqn5btgKKC5yfCvg/details}
}
"""
"""
This code containts functions to take in raw event records and produce endogenous influence, exogenous influence and message files 
for the initialization of the Multi-Action Cascade Model
If you find bugs or make fixes please communicate to chathikagunaratne@gmail.com 
"""

import pandas as pd
from numba import cuda, jit
import argparse
import math
import numpy as np
import datetime as dt
import pickle
import time
import os


ACTIVITY_THRESHOLD = {'twitter': 17, 'youtube': 2}

ACTION_MAP = {
    "creation": ["CreateEvent","tweet","post","Post","video"],
    "contribution": ['IssueCommentEvent', 'PullRequestEvent',
    'GollumEvent', 'PullRequestReviewCommentEvent', 'PushEvent', 
    'IssuesEvent', 'CommitCommentEvent',"DeleteEvent","reply","quote","message","comment","Comment"],
    "sharing": ["ForkEvent","WatchEvent", 'ReleaseEvent', 'MemberEvent', 'PublicEvent',"retweet"]
}

def getEventTypeIdx(event):    
    """
    """
    for idx,name in enumerate(ACTION_MAP.keys()):
        if event in ACTION_MAP[name]:
            return idx    

def ensurePlatformUniqueness(events):
    events=events.copy()
    events.loc[events.parentID.isna(),"parentID"]=events.loc[events.parentID.isna(),"nodeID"]
    events.loc[events.conversationID.isna(),"conversationID"]=events.loc[events.conversationID.isna(),"nodeID"]
    if("platform" in events.columns):
        events.loc[:,"userID"] = events.apply(lambda x: str(x.platform) + "_" + str(x.userID),axis=1)
        events.loc[:,"nodeID"] = events.apply(lambda x: str(x.platform) + "_" + str(x.nodeID),axis=1)
        events.loc[:,"parentID"] = events.apply(lambda x: str(x.platform) + "_" + str(x.parentID),axis=1)
        events.loc[:,"conversationID"] = events.apply(lambda x: str(x.platform) + "_" + str(x.conversationID),axis=1)    
    return events

def numerifyEvents(events):
    """
    Sorts users in events in alphabetical order
    assigns numbering
    returns modified events and user mapping
    umapping: pandas dataframe of user names where names are columns (using this structure since it allows O(1) time access to user names)
    """
    events = ensurePlatformUniqueness(events)
    print(events.head())
    events = events[events.userID.notnull() & events.nodeID.notnull()]# & events.parentID.isnull() & events.conversationID.isnull()] 
    userlist = events.userID.unique()
    umapping = pd.Series(np.sort(list(set(userlist)))).reset_index().set_index(0).T
    events["userID"] = [ umapping[x][0] for x in events["userID"]]

    targetlist = events.nodeID.append(events.parentID).append(events.conversationID).unique()
    tmapping = pd.Series(np.sort(list(set(targetlist)))).reset_index().set_index(0).T

    events["nodeID"] = [ tmapping[x][0] for x in events["nodeID"]]
    #print(events.nodeID.unique())
    #events["parentID"] = [ tmapping[x][0] for x in events["parentID"]]
    #events["conversationID"] = [ tmapping[x][0] for x in events["conversationID"]]
    events["action"] = events["action"].apply(lambda x: getEventTypeIdx(x))
    return (events,umapping,tmapping)

def numerifyShocks(shocks_):
    """
    Sorts shocks in alphabetical order
    assigns numbering
    returns modified shocks mapping
    """
    shocks = shocks_.copy()
    shocklist = list(shocks.drop("time",axis=1).columns)
    smapping = pd.Series(np.sort(list(set(shocklist)))).reset_index().set_index(0).T
    shocks = shocks.set_index("time")
    shocks.columns = [ smapping[col][0] for col in shocks.columns]
    shocks = shocks.reset_index()
    return (shocks, smapping)
    
@cuda.jit()
def calcUserH(events_matrix,H):
    """
    Shannon entropy calculated on all users. Units dits.
    """
    userID = cuda.grid(1)
    if userID >= events_matrix.shape[0]:
        return
    userID = int(userID)
    p_1 = 0.0
    p_0 = 0.0    
    for i in range(events_matrix.shape[1]):
        if events_matrix[userID,i] > 0:
            p_1 = p_1 + 1
        if events_matrix[userID,i] == 0:
            p_0 = p_0 + 1
    p_0 = float(p_0 / int(events_matrix.shape[1]))
    p_1 = float(p_1 / int(events_matrix.shape[1]))
    h = float(-1 * (p_1 * math.log(p_1)  + p_0 * math.log(p_0) ))
    H[userID] = h
    #print(events[userID][0])

@cuda.jit()
def calcProb1(events_matrix,Prob1):
    """
    Probability of a user performing an action per hour.
    """
    userID = cuda.grid(1)
    if userID >= events_matrix.shape[0]:
        return
    userID = int(userID)
    p_1 = 0.0   
    for i in range(events_matrix.shape[1]):
        if events_matrix[userID,i] > 0:
            p_1 = p_1 + 1
    p_1 = float(p_1 / int(events_matrix.shape[1]))
    Prob1[userID] = p_1

@cuda.jit()
def calcPartialH(events_matrix,H):
    """
    Shannon entropy calculated on all users. Units dits.
    """
    userID = cuda.grid(1)
    if userID >= events_matrix.shape[0]:
        return
    userID = int(userID)
    p_1 = 0.0   
    for i in range(events_matrix.shape[1]):
        if events_matrix[userID,i] > 0:
            p_1 = p_1 + 1
    p_1 = float(p_1 / int(events_matrix.shape[1]))
    h = float(-1 * (p_1 * math.log(p_1) ))
    H[userID] = h
    #print(events[userID][0])


@cuda.jit()
def calcT(influencer_events_matrix,influencee_events_matrix,T):
    """
    Transfer entropy calculated on all relationships. Units dits.
    """
    influencerID, influenceeID  = cuda.grid(2)
    if influencerID >= influencer_events_matrix.shape[0] or influenceeID >= influencee_events_matrix.shape[0]:
        return
    influencerID = int(influencerID)
    influenceeID = int(influenceeID)
    #Calculate destination conditioned on past probabilities
    dest0_condition_past0_count = 0    
    dest1_condition_past0_count = 0
    dest0_condition_past1_count = 0
    dest1_condition_past1_count = 0
    past0_count = 0
    past1_count = 0
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencee_events_matrix[influenceeID,i] == 0:
            past0_count = past0_count + 1
            if influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_past0_count = dest0_condition_past0_count + 1
            elif influencee_events_matrix[influenceeID,i+1] > 0:
                dest1_condition_past0_count = dest1_condition_past0_count + 1
        if influencee_events_matrix[influenceeID,i] == 1:
            past1_count = past1_count + 1
            if influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_past1_count = dest0_condition_past1_count + 1
            elif influencee_events_matrix[influenceeID,i+1] > 0:
                dest1_condition_past1_count = dest1_condition_past1_count + 1
    #Calculate destination _conditioned on past and source probabilities
    dest0_condition_source0_past0_count = 0
    dest1_condition_source0_past0_count = 0
    dest0_condition_source0_past1_count = 0
    dest1_condition_source0_past1_count = 0
    dest0_condition_source1_past0_count = 0
    dest1_condition_source1_past0_count = 0
    dest0_condition_source1_past1_count = 0
    dest1_condition_source1_past1_count = 0 
    source0_past0_count = 0
    source0_past1_count = 0
    source1_past0_count = 0
    source1_past1_count = 0    
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencer_events_matrix[influencerID,i] == 0 and influencee_events_matrix[influenceeID,i] == 0:
            source0_past0_count = source0_past0_count + 1
            if   influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_source0_past0_count = dest0_condition_source0_past0_count + 1
            elif influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source0_past0_count = dest1_condition_source0_past0_count + 1
        if influencer_events_matrix[influencerID,i] == 0 and influencee_events_matrix[influenceeID,i] > 0:
            source0_past1_count = source0_past1_count + 1
            if   influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_source0_past1_count = dest0_condition_source0_past1_count + 1
            elif influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source0_past1_count = dest1_condition_source0_past1_count + 1
        if influencer_events_matrix[influencerID,i] >  0 and influencee_events_matrix[influenceeID,i] == 0:
            source1_past0_count = source1_past0_count + 1
            if   influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_source1_past0_count = dest0_condition_source1_past0_count + 1
            elif influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source1_past0_count = dest1_condition_source1_past0_count + 1
        if influencer_events_matrix[influencerID,i] >  0 and influencee_events_matrix[influenceeID,i] > 0:
            source1_past1_count = source1_past1_count + 1
            if   influencee_events_matrix[influenceeID,i+1] == 0:
                dest0_condition_source1_past1_count = dest0_condition_source1_past1_count + 1
            elif influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source1_past1_count = dest1_condition_source1_past1_count + 1
    TE =  dest0_condition_source0_past0_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest0_condition_source0_past0_count / source0_past0_count) / (dest0_condition_past0_count / past0_count)) \
        + dest1_condition_source0_past0_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source0_past0_count / source0_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest0_condition_source0_past1_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest0_condition_source0_past1_count / source0_past1_count) / (dest0_condition_past1_count / past1_count)) \
        + dest1_condition_source0_past1_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source0_past1_count / source0_past1_count) / (dest1_condition_past1_count / past1_count)) \
        + dest0_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest0_condition_source1_past0_count / source1_past0_count) / (dest0_condition_past0_count / past0_count)) \
        + dest1_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source1_past0_count / source1_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest0_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest0_condition_source1_past1_count / source1_past1_count) / (dest0_condition_past1_count / past1_count)) \
        + dest1_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source1_past1_count / source1_past1_count) / (dest1_condition_past1_count / past1_count))

    T[(influencerID*influencee_events_matrix.shape[0]) + influenceeID] = TE

@cuda.jit()
def calcPartialT(influencer_events_matrix,influencee_events_matrix,partialT):
    """
    Partial transfer entropy calculated on all relationships. Units dits.
    """
    influencerID, influenceeID  = cuda.grid(2)
    if influencerID >= influencer_events_matrix.shape[0] or influenceeID >= influencee_events_matrix.shape[0]:
        return
    influencerID = int(influencerID)
    influenceeID = int(influenceeID)
    #Calculate destination conditioned on past probabilities
    dest1_condition_past0_count = 0
    dest1_condition_past1_count = 0
    past0_count = 0
    past1_count = 0
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencee_events_matrix[influenceeID,i] == 0:
            past0_count = past0_count + 1
            if influencee_events_matrix[influenceeID,i+1] > 0:
                dest1_condition_past0_count = dest1_condition_past0_count + 1
        if influencee_events_matrix[influenceeID,i] == 1:
            past1_count = past1_count + 1
            if influencee_events_matrix[influenceeID,i+1] > 0:
                dest1_condition_past1_count = dest1_condition_past1_count + 1
    #Calculate destination _conditioned on past and source probabilities
    dest1_condition_source1_past0_count = 0
    dest1_condition_source1_past1_count = 0 
    source1_past0_count = 0
    source1_past1_count = 0    
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencer_events_matrix[influencerID,i] >  0 and influencee_events_matrix[influenceeID,i] == 0:
            source1_past0_count = source1_past0_count + 1
            if influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source1_past0_count = dest1_condition_source1_past0_count + 1
        if influencer_events_matrix[influencerID,i] >  0 and influencee_events_matrix[influenceeID,i] > 0:
            source1_past1_count = source1_past1_count + 1
            if influencee_events_matrix[influenceeID,i+1] >  0:
                dest1_condition_source1_past1_count = dest1_condition_source1_past1_count + 1
    partial_t = dest1_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source1_past0_count / source1_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest1_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log( (dest1_condition_source1_past1_count / source1_past1_count) / (dest1_condition_past1_count / past1_count))

    partialT[(influencerID*influencee_events_matrix.shape[0]) + influenceeID] = partial_t

@cuda.jit()
def cudaResample(compressed_events,resampled_events):
    userID = int(cuda.grid(1))
    if userID > compressed_events.shape[0]:
        return
    for i in range(compressed_events.shape[1]):
        time_delta = int(math.floor(compressed_events[userID,i]))
        if time_delta == -1:
            break
        resampled_events[userID,time_delta] = resampled_events[userID,time_delta] + 1

def extractInfoIDProbDists(in_events, in_umapping):
    print('Calculating contentID probabilities.')
    start_time = time.time()
    print(in_events.columns)
    # make node_to_infoidList mapping
    node_to_infoidList = in_events.set_index('conversationID').to_dict()['informationIDs']
    node_to_infoidList.update(in_events.set_index('parentID').to_dict()['informationIDs'])
    node_to_infoidList.update(in_events.set_index('nodeID').to_dict()['informationIDs'])
    for k in node_to_infoidList.keys():
        node_to_infoidList[k] = eval(node_to_infoidList[k])
    # get a list of all unique informationIDs
    allInfoIds = set()
    for infoidlist in in_events.informationIDs:
        allInfoIds.update(eval(infoidlist))
    # create mapping of unique infoid to index
    infoid_to_index = {}
    tempIdx = 0
    for infoid in allInfoIds:
        infoid_to_index[infoid] = tempIdx
        tempIdx += 1
    # general counters
    numOfInfoIDs = len(infoid_to_index)
    countCond = np.zeros((numOfInfoIDs, numOfInfoIDs))
    countBase = np.zeros(numOfInfoIDs)
    # user based counters
    numOfUsers = in_umapping.shape[1]
    userCountCond = np.zeros((numOfUsers, numOfInfoIDs, numOfInfoIDs))
    userCountBase = np.zeros((numOfUsers, numOfInfoIDs))
    # counting process
    for idx, row in in_events.iterrows():
        if row['parentID'] != row['nodeID']:
            for pnar in node_to_infoidList[ row['parentID'] ]:
                for nar in node_to_infoidList[ row['nodeID'] ]:
                    pnaridx = infoid_to_index[pnar]
                    countBase[pnaridx] += 1.0
                    countCond[pnaridx, infoid_to_index[nar]] += 1.0
                    userCountBase[ row['userID'], pnaridx ] += 1.0
                    userCountCond[ row['userID'], pnaridx, infoid_to_index[nar] ] += 1.0

    probs = np.zeros((numOfInfoIDs,numOfInfoIDs))
    for parentNar in range(numOfInfoIDs):
        for childNar in range(numOfInfoIDs):
            if countBase[parentNar] != 0.0:
                probs[parentNar, childNar] = countCond[parentNar, childNar] / countBase[parentNar]

    probsUser = np.zeros((numOfUsers, numOfInfoIDs, numOfInfoIDs))
    for usr in range(numOfUsers):
        for parentNar in range(numOfInfoIDs):
            for childNar in range(numOfInfoIDs):
                if userCountBase[usr, parentNar] != 0.0:
                    probsUser[usr, parentNar, childNar] = userCountCond[usr, parentNar, childNar] / userCountBase[usr, parentNar]
    
    # get the list of infoIDs in the order of its index in this code
    orderedInfoids = [i for i in range(numOfInfoIDs)]
    for k in infoid_to_index.keys():
        orderedInfoids[infoid_to_index[k]] = k
    
    dfprobs = pd.DataFrame(data=probs, index=orderedInfoids, columns=orderedInfoids)
    dfprobs.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_ContentIDProbDists.csv",index_label='parentNarrative')

    cols = [list(in_umapping.columns), orderedInfoids, orderedInfoids]
    mi = pd.MultiIndex.from_product(cols, names=['userID','parentInfoID','childInfoID'])
    dfprobsUser = pd.Series(index=mi, data=probsUser.flatten())
    dfprobsUser = dfprobsUser[dfprobsUser > 0.0]
    dfprobsUser.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_UserBasedContentIDProbDists.csv", header=['probVals'])
    print('Done calculating contentID probabilities. Time taken: {}'.format(time.time() - start_time))

def extractEndogenousInfluence(all_events, u, t):
    """
    Start by assuming fully connected network
    make n*n matrix (n=number of users),  where each cell contains two activity timeseries
    rows = influencer
    cols = influencee
    """
    average_H = pd.DataFrame()
    average_Prob1 = pd.DataFrame()
    average_partialH = pd.DataFrame()
    average_T = pd.DataFrame(index = pd.MultiIndex.from_product([list(range(u.shape[1])),list(range(u.shape[1]))], names=["userID0", "userID1"])) 
    average_partialT = pd.DataFrame(index = pd.MultiIndex.from_product([list(range(u.shape[1])),list(range(u.shape[1]))], names=["userID0", "userID1"])) 
    num_days = math.floor((all_events.time.max() - all_events.time.min()).total_seconds()/86400)
    step_size = 100
    for day_i in range(0,max(step_size,num_days-step_size),step_size):
        period_start = all_events.time.min() + dt.timedelta(days = day_i)
        period_end = all_events.time.min() + dt.timedelta(days = day_i + step_size)
        print(period_start)       
        print(period_end)
        print(range(0,step_size,num_days-step_size))
        events = all_events[(all_events.time > period_start) & (all_events.time < period_end)].copy()
        if events.shape[0] == 0:
            break
        print("Generating event matrix.")
        start = time.time()
        time_res = "H"
        time_max = events.time.max()
        time_min = events.time.min()
        #Ensure min and max timestamps, then resample and count
        #cudafy data
        events.loc[:,"time"] = events.time.apply(lambda x: int((x - time_min).total_seconds()//3600)) # get time as hours float
        events = events[["userID","action","time"]].sort_values(["userID","action"])
        max_events_per_user_action = events.groupby(["userID","action"]).apply(lambda x: x.shape[0]).max() # max user_action count
        events_matrix = np.zeros((u.shape[1],len(list(ACTION_MAP.keys())),max_events_per_user_action))
        for action in range(len(list(ACTION_MAP.keys()))):
            events_this_action = events[events["action"]==action]
            max_events_by_user_this_action = events_this_action.groupby("userID").apply(lambda x: x.shape[0]).max()
            try:
                compressed_events = np.full((u.shape[1],max_events_by_user_this_action+1),-1.0)        
            except:
                print("No events!")
                print(max_events_by_user_this_action)
            for userID in range(u.shape[1]):
                events_by_user_this_action = events_this_action[events_this_action["userID"] == userID]
                for i, event_time in enumerate(events_by_user_this_action.time):
                    compressed_events[userID,i] = event_time
            compressed_events = cuda.to_device(compressed_events)        
            events_matrix_this_action = cuda.to_device(np.zeros((u.shape[1],max_events_per_user_action)))
            bpg, tpb = _gpu_init_1d(u.shape[1])
            cudaResample[bpg,tpb](compressed_events,events_matrix_this_action)
            events_matrix[:,action,:] = np.nan_to_num(events_matrix_this_action.copy_to_host())
            print(np.sum(events_matrix[:,action,:]))
            print("Resampled and matrixified " + str(list(ACTION_MAP.keys())[action]) + " events on GPU.")
            #compressed_events = None
            #events_matrix_this_action = None
        print(np.sum(events_matrix,axis=1))
        ###########################################################################################################
        #Calculate entropy per action
        all_H = np.zeros((u.shape[1],len(list(ACTION_MAP.keys())) ),dtype=np.float64)
        all_Prob1 = np.zeros((u.shape[1],len(list(ACTION_MAP.keys())) ),dtype=np.float64)
        all_partialH = np.zeros((u.shape[1],len(list(ACTION_MAP.keys())) ),dtype=np.float64)
        end = time.time()
        print("GPU took " + str(end-start) + " seconds to resample and matrixify event data.")    
        for action in range(len(list(ACTION_MAP.keys()))):
            events_this_action = cuda.to_device(np.ascontiguousarray(events_matrix[:,action,:]))
            #Start cuda calculations of H
            print(events_this_action.shape[0])
            bpg, tpb = _gpu_init_1d(events_this_action.shape[0])
            H = cuda.to_device(np.zeros(events_this_action.shape[0],dtype=np.float64))
            start = time.time()
            calcUserH[bpg, tpb](events_this_action,H)
            end = time.time()
            all_H[:,action] = H.copy_to_host().tolist()
            print("Time taken for entropy calculations through CUDA: " + str(end-start) + " seconds.")       
            prob1 = cuda.to_device(np.zeros(events_this_action.shape[0],dtype=np.float64))
            start = time.time()
            calcProb1[bpg, tpb](events_this_action,prob1)
            end = time.time()
            all_Prob1[:,action] = prob1.copy_to_host().tolist()
            print("Time taken for hourly active probability calculations through CUDA: " + str(end-start) + " seconds.")  
            partialH = cuda.to_device(np.zeros(events_this_action.shape[0],dtype=np.float64))
            start = time.time()
            calcPartialH[bpg, tpb](events_this_action,partialH)
            end = time.time()
            all_partialH[:,action] = partialH.copy_to_host().tolist()
            print("Time taken for partial entropy calculations through CUDA: " + str(end-start) + " seconds.")         
        #all_H.index = all_H["userID0"].apply(lambda x: u.columns[x])
        all_H = pd.DataFrame(all_H,columns = list(ACTION_MAP.keys())).fillna(0)
        all_H.index = all_H.index.set_names(['userID'])
        all_H = all_H.reset_index()
        print(all_H.head())
        all_H["userID"] = all_H["userID"].apply(lambda x: u.columns[x])
        print("Entropy calculations done.")
        all_H = all_H.set_index(["userID"])
        take_mean = lambda s1, s2: (s1 + s2) / 2
        average_H = average_H.combine(all_H,func=take_mean,fill_value=0)
        average_H.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_Entropy.csv",index=True)
        ###
        all_Prob1 = pd.DataFrame(all_Prob1,columns = list(ACTION_MAP.keys())).fillna(0)
        all_Prob1.index = all_Prob1.index.set_names(['userID'])
        all_Prob1 = all_Prob1.reset_index()
        print(all_Prob1.head())
        all_Prob1["userID"] = all_Prob1["userID"].apply(lambda x: u.columns[x])
        print("Entropy calculations done.")
        all_Prob1 = all_Prob1.set_index(["userID"])
        take_mean = lambda s1, s2: (s1 + s2) / 2
        average_Prob1 = average_Prob1.combine(all_Prob1,func=take_mean,fill_value=0)
        average_Prob1.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_Hourly_Activity_Probability.csv",index=True)
        ###
        all_partialH = pd.DataFrame(all_partialH,columns = list(ACTION_MAP.keys())).fillna(0)
        all_partialH.index = all_partialH.index.set_names(['userID'])
        all_partialH = all_partialH.reset_index()
        print(all_partialH.head())
        all_partialH["userID"] = all_partialH["userID"].apply(lambda x: u.columns[x])
        print("Entropy calculations done.")
        all_partialH = all_partialH.set_index(["userID"])
        take_mean = lambda s1, s2: (s1 + s2) / 2
        average_partialH = average_partialH.combine(all_partialH,func=take_mean,fill_value=0)
        average_partialH.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_Partial_Entropy.csv",index=True)
        ###########################################################################################################
        #Calculate Transfer Entropy per action->action relationship
        te_start = time.time()
        all_T = pd.DataFrame()
        all_partialT = pd.DataFrame()
        for influencer_action in range(len(list(ACTION_MAP.keys()))):
            events_influencer_action = cuda.to_device(np.ascontiguousarray(events_matrix[:,influencer_action,:]))
            for influencee_action in range(len(list(ACTION_MAP.keys()))):
                start = time.time()
                events_influencee_action = cuda.to_device(np.ascontiguousarray(events_matrix[:,influencee_action,:]))
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for cuda.to_device(np.ascontiguousarray(events_matrix[:,influencee_action,:])).")

                start = time.time()
                #Start cuda calculations of T
                bpg, tpb = _gpu_init_2d(events_influencer_action.shape[0],events_influencee_action.shape[0])
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for _gpu_init_2d(events_influencer_action.shape[0],events_influencee_action.shape[0]).")

                start = time.time()
                T = np.zeros((events_influencer_action.shape[0]*events_influencee_action.shape[0]),dtype=np.float32)
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for np.zeros((events_influencer_action.shape[0]*events_influencee_action.shape[0]),dtype=np.float32).")

                start = time.time()
                T = cuda.to_device(T)
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for cuda.to_device(T).")

                start = time.time()
                calcT[bpg, tpb](events_influencer_action,events_influencee_action,T)
                end = time.time()
                print("GPU took " + str(end-start) + " seconds for transfer entropy calculations through CUDA.")

                start = time.time()
                relationship_name = list(ACTION_MAP.keys())[influencer_action] + "To" + list(ACTION_MAP.keys())[influencee_action]
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for list(ACTION_MAP.keys())....")

                start = time.time()
                T = pd.DataFrame(T.copy_to_host().tolist(),columns = [relationship_name], index = pd.MultiIndex.from_product([list(range(u.shape[1])),list(range(u.shape[1]))], names=["userID0", "userID1"]))
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for pd.DataFrame(T.copy_to_host().tolist(),....")

                start = time.time()
                if all_T.empty:
                    all_T = T
                else:
                    all_T = all_T.join(T,how="outer")                
                #Start cuda calculations of partialT
                partialT = np.zeros((events_influencer_action.shape[0]*events_influencee_action.shape[0]),dtype=np.float32)
                partialT = cuda.to_device(partialT)
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for if all_T:......")

                start = time.time()
                calcPartialT[bpg, tpb](events_influencer_action,events_influencee_action,partialT)
                end = time.time()
                print("GPU took " + str(end-start) + " seconds for partial transfer entropy calculations through CUDA.")
                
                start = time.time()
                partialT = pd.DataFrame(partialT.copy_to_host().tolist(),columns = [relationship_name], index = pd.MultiIndex.from_product([list(range(u.shape[1])),list(range(u.shape[1]))], names=["userID0", "userID1"]))
                if all_partialT.empty:
                    all_partialT = partialT
                else:
                    all_partialT = all_partialT.join(partialT,how="outer")
                end = time.time()
                print("CPU took " + str(end-start) + " seconds for final step....")
                print("Transfer entropy for relationship " + relationship_name + " done.")
                #events_influencee_action = None
                #T = None
        #write T
        all_T = all_T.reset_index()
        all_T = all_T.fillna(0.)
        all_T = all_T.set_index(["userID0","userID1"])
        average_T = average_T.combine(all_T,func=take_mean,fill_value=0)
        average_T_out = average_T.copy()
        #average_T_out = average_T_out[average_T_out.iloc[:,2:].sum(axis=1) > 0] commented to avoid losing users with no social influence
        average_T_out = average_T_out.reset_index()
        average_T_out["userID0"] = average_T_out["userID0"].apply(lambda x: u.columns[x])
        average_T_out["userID1"] = average_T_out["userID1"].apply(lambda x: u.columns[x])        
        average_T_out.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_Transfer_Entropy.csv",index = False)
        #write partial T
        all_partialT = all_partialT.reset_index()
        all_partialT = all_partialT.fillna(0.)
        all_partialT = all_partialT.set_index(["userID0","userID1"])
        average_partialT = average_partialT.combine(all_T,func=take_mean,fill_value=0)
        average_partialT_out = average_partialT.copy()
        #average_partialT_out = average_partialT_out[average_partialT_out.iloc[:,2:].sum(axis=1) > 0] commented to avoid losing users with no social influence
        average_partialT_out = average_partialT_out.reset_index()
        average_partialT_out["userID0"] = average_partialT_out["userID0"].apply(lambda x: u.columns[x])
        average_partialT_out["userID1"] = average_partialT_out["userID1"].apply(lambda x: u.columns[x])        
        average_partialT_out.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Endogenous_Partial_Transfer_Entropy.csv",index = False)        
        print("Percent complete =" + str(max(100,100*(day_i+7)/num_days)) + "%")        
    te_end = time.time()
    
    print("Took " + str(te_end - te_start) + " seconds to calculate all transfer entropies.")
    return average_T_out

############################ Exogenous Extraction #############################################
def extractExogenousInfluence(all_events,u, t, all_shocks):
    """
    Start by assuming fully connected network
    make n*n matrix (n=number of users),  where each cell contains two activity timeseries
    rows = influencer
    cols = influencee
    """
    print("Numerifying Shocks.")
    start = time.time()
    all_shocks, s = numerifyShocks(all_shocks)
    print("Time taken to numerify shock data: " + str(time.time()-start) + " seconds.")
    del start
    average_H = pd.DataFrame()
    average_partialH = pd.DataFrame()
    average_T = pd.DataFrame(index = pd.MultiIndex.from_product([list(range(s.shape[1])),list(range(u.shape[1]))], names=["shock", "userID"])) 
    average_partialT = pd.DataFrame(index = pd.MultiIndex.from_product([list(range(s.shape[1])),list(range(u.shape[1]))], names=["shock", "userID"])) 
    num_days = math.floor((all_events.time.max() - all_events.time.min()).total_seconds()/86400)
    step_size = 100
    for day_i in range(0,max(step_size,num_days-step_size),step_size):
        period_start = all_events.time.min() + dt.timedelta(days = day_i)
        period_end = all_events.time.min() + dt.timedelta(days = day_i + step_size)
        print(period_start)       
        print(period_end)
        print(range(0,step_size,num_days-step_size))
        events = all_events[(all_events.time > period_start) & (all_events.time < period_end)].copy()
        shocks = all_shocks[(all_shocks.time > period_start) & (all_shocks.time < period_end)].copy()
        if events.shape[0] == 0 or shocks.shape[0] == 0:
            break
        print("Generating event matrix.")
        start = time.time()
        time_res = "H"
        time_max = events.time.max()
        time_min = events.time.min()
        #Ensure min and max timestamps, then resample and count
        #cudafy data
        events.loc[:,"time"] = events.time.apply(lambda x: int((x - time_min).total_seconds()//3600))
        shocks.loc[:,"time"] = shocks.time.apply(lambda x: int((x - time_min).total_seconds()//3600))
        events = events[["userID","action","time"]].sort_values(["userID","action"])
        shocks = shocks.set_index("time")
        shocks = shocks[sorted(shocks.columns)]
        #Cudafy shocks
        max_events_per_user_action = events.groupby(["userID","action"]).apply(lambda x: x.shape[0]).max()
        max_times_shock_occurred = int(shocks.sum().max())
        compressed_shocks = np.full((s.shape[1],max_times_shock_occurred+1),-1.0)        
        for shockID in range(s.shape[1]):
            times_this_shock_occurred = shocks[[shockID]].copy()
            times_this_shock_occurred = times_this_shock_occurred[times_this_shock_occurred[shockID]>0].reset_index()
            for i, shock_time in enumerate(times_this_shock_occurred.time):
                compressed_shocks[shockID,i] = shock_time
        compressed_shocks = cuda.to_device(compressed_shocks)      
        shocks_matrix = cuda.to_device(np.zeros((s.shape[1],max_events_per_user_action)))
        bpg, tpb = _gpu_init_1d(s.shape[1])
        cudaResample[bpg,tpb](compressed_shocks,shocks_matrix)
        #Now do events        
        events_matrix = np.zeros((u.shape[1],len(list(ACTION_MAP.keys())),max_events_per_user_action))
        for action in range(len(list(ACTION_MAP.keys()))):
            events_this_action = events[events["action"]==action]
            max_events_by_user_this_action = events_this_action.groupby("userID").apply(lambda x: x.shape[0]).max()
            compressed_events = np.full((u.shape[1],max_events_by_user_this_action+1),-1.0)        
            for userID in range(u.shape[1]):
                events_by_user_this_action = events_this_action[events_this_action["userID"] == userID]
                for i, event_time in enumerate(events_by_user_this_action.time):
                    compressed_events[userID,i] = event_time
            compressed_events = cuda.to_device(compressed_events)        
            events_matrix_this_action = cuda.to_device(np.zeros((u.shape[1],max_events_per_user_action)))
            bpg, tpb = _gpu_init_1d(u.shape[1])
            cudaResample[bpg,tpb](compressed_events,events_matrix_this_action)
            events_matrix[:,action,:] = np.nan_to_num(events_matrix_this_action.copy_to_host())
            print(np.sum(events_matrix[:,action,:]))
            print("Resampled and matrixified " + str(list(ACTION_MAP.keys())[action]) + " events on GPU.")
            #compressed_events = None
            #events_matrix_this_action = None
        #print(np.sum(events_matrix,axis=1))
        ###########################################################################################################
        #Calculate entropy per shock
        all_H = np.zeros(s.shape[1],dtype=np.float64)
        partial_H = np.zeros(s.shape[1],dtype=np.float64)
        end = time.time()
        print("GPU took " + str(end-start) + " seconds to resample and matrixify event data.")    
        #events_this_action = cuda.to_device(np.ascontiguousarray(shocks_matrix))
        #Start cuda calculations of H
        print(shocks_matrix.shape[0])
        bpg, tpb = _gpu_init_1d(shocks_matrix.shape[0])
        H = cuda.to_device(np.zeros(shocks_matrix.shape[0],dtype=np.float64))
        start = time.time()
        calcUserH[bpg, tpb](shocks_matrix,H)
        end = time.time()
        all_H = H.copy_to_host().tolist()
        print("Time taken for entropy calculations through CUDA: " + str(end-start) + " seconds.")         
        ###
        partialH = cuda.to_device(np.zeros(shocks_matrix.shape[0],dtype=np.float64))
        start = time.time()
        calcPartialH[bpg, tpb](shocks_matrix,partialH)
        end = time.time()
        all_partialH = partialH.copy_to_host().tolist()
        print("Time taken for partial entropy calculations through CUDA: " + str(end-start) + " seconds.")         
        #all_H.index = all_H["userID0"].apply(lambda x: u.columns[x])
        all_H = pd.DataFrame(all_H,columns = ["H"]).fillna(0)
        all_H.index = all_H.index.set_names(['shockID'])
        all_H = all_H.reset_index()
        print(all_H.head())
        all_H["shockID"] = all_H["shockID"].apply(lambda x: s.columns[x])
        print("Entropy calculations done.")
        all_H = all_H.set_index(["shockID"])
        take_mean = lambda s1, s2: (s1 + s2) / 2
        average_H = average_H.combine(all_H,func=take_mean,fill_value=0)
        average_H.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Exogenous_Entropy.csv",index=True)
        ###
        all_partialH = pd.DataFrame(all_partialH,columns = ["H"]).fillna(0)
        all_partialH.index = all_partialH.index.set_names(['shockID'])
        all_partialH = all_partialH.reset_index()
        print(all_partialH.head())
        all_partialH["shockID"] = all_partialH["shockID"].apply(lambda x: s.columns[x])
        print("Entropy calculations done.")
        all_partialH = all_partialH.set_index(["shockID"])
        take_mean = lambda s1, s2: (s1 + s2) / 2
        average_partialH = average_partialH.combine(all_partialH,func=take_mean,fill_value=0)
        average_partialH.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Exogenous_Partial_Entropy.csv",index=True)
        ###########################################################################################################
        #Calculate Transfer Entropy per action->action relationship
        te_start = time.time()
        all_T = pd.DataFrame()
        all_partialT = pd.DataFrame()
        for influencee_action in range(len(list(ACTION_MAP.keys()))):
            events_influencee_action = cuda.to_device(np.ascontiguousarray(events_matrix[:,influencee_action,:]))
            #Start cuda calculations of T
            bpg, tpb = _gpu_init_2d(shocks_matrix.shape[0],events_influencee_action.shape[0])
            T = np.zeros((shocks_matrix.shape[0]*events_influencee_action.shape[0]),dtype=np.float32)
            T = cuda.to_device(T)
            start = time.time()
            calcT[bpg, tpb](shocks_matrix,events_influencee_action,T)
            end = time.time()
            print("GPU took " + str(end-start) + " seconds for transfer entropy calculations through CUDA.")
            relationship_name = list(ACTION_MAP.keys())[influencee_action]
            T = pd.DataFrame(T.copy_to_host().tolist(),columns = [relationship_name], index = pd.MultiIndex.from_product([list(range(s.shape[1])),list(range(u.shape[1]))], names=["shockID", "userID"]))
            if all_T.empty:
                all_T = T
            else:
                all_T = all_T.join(T,how="outer")                
            #Start cuda calculations of partialT
            partialT = np.zeros((shocks_matrix.shape[0]*events_influencee_action.shape[0]),dtype=np.float32)
            partialT = cuda.to_device(partialT)
            start = time.time()
            calcPartialT[bpg, tpb](shocks_matrix,events_influencee_action,partialT)
            end = time.time()
            print("GPU took " + str(end-start) + " seconds for partial transfer entropy calculations through CUDA.")
            partialT = pd.DataFrame(partialT.copy_to_host().tolist(),columns = [relationship_name], index = pd.MultiIndex.from_product([list(range(s.shape[1])),list(range(u.shape[1]))], names=["shockID", "userID"]))
            if all_partialT.empty:
                all_partialT = partialT
            else:
                all_partialT = all_partialT.join(partialT,how="outer")
            print("Transfer entropy for relationship " + relationship_name + " done.")
            #events_influencee_action = None
            #T = None
        #write T
        all_T = all_T.reset_index()
        all_T = all_T.fillna(0.)
        all_T = all_T.set_index(["shockID", "userID"])
        print(all_T)
        print(all_T.sum())
        average_T = average_T.combine(all_T,func=take_mean,fill_value=0)
        average_T_out = average_T.copy()
        average_T_out = average_T_out.reset_index()
        #average_T_out = average_T_out[average_T_out.iloc[:,2:].sum(axis=1) > 0]     commented to avoid losing users with no social influence    
        average_T_out["shockID"] = average_T_out["shockID"].apply(lambda x: s.columns[x])
        average_T_out["userID"] = average_T_out["userID"].apply(lambda x: u.columns[x])        
        average_T_out.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Exogenous_Transfer_Entropy.csv",index = False)
        #write partial T
        all_partialT = all_partialT.reset_index()
        all_partialT = all_partialT.fillna(0.)
        all_partialT = all_partialT.set_index(["shockID","userID"])
        average_partialT = average_partialT.combine(all_T,func=take_mean,fill_value=0)
        average_partialT_out = average_partialT.copy()
        average_partialT_out = average_partialT_out.reset_index()
        #average_partialT_out = average_partialT_out[average_partialT_out.iloc[:,2:].sum(axis=1) > 0]     commented to avoid losing users with no social influence    
        average_partialT_out["shockID"] = average_partialT_out["shockID"].apply(lambda x: s.columns[x])
        average_partialT_out["userID"] = average_partialT_out["userID"].apply(lambda x: u.columns[x])        
        average_partialT_out.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Exogenous_Partial_Transfer_Entropy.csv",index = False)        
        print("Percent complete =" + str(max(100,100*(day_i+7)/num_days)) + "%")        
    te_end = time.time()
    print("Took " + str(te_end - te_start) + " seconds to calculate all transfer entropies.")

############################ Message Extraction #############################################
#### Worker function. Takes some users and finds the last n messages that they have received.
def extractMessagesFromChunk(influencedUsers):
    messages = []
    for influencedUser in influencedUsers:                
        influencersOfInfluencedUser = network[network["userID1"]==influencedUser]["userID0"]
        eventsByInfluencers = gEvents[gEvents["userID"].isin(influencersOfInfluencedUser.tolist())].sort_values(by="time", ascending=False)
        for idx, event in eventsByInfluencers.iloc[:100,:].iterrows():
            messages.append(event.values)
    return messages
network = None
gEvents = None
##### Master function, splits the influenced user list and then asks workers to find their last n received messages.
import multiprocessing
def extractMessages(eventsfile, network_dataframe):
    print("Extracting Messages")
    global gEvents
    gEvents = pd.read_csv(eventsfile,parse_dates=['time'])[['userID', 'nodeID', 'parentID', 'conversationID', 'time', 'action', 'platform', 'informationIDs']].copy()
    gEvents = ensurePlatformUniqueness(gEvents)
    global network
    network = network_dataframe
    print('file reading done.')
    print('network:')
    print(network)
    print(network.columns)
    print('events:')
    print(gEvents.columns)
    print(gEvents)
    network =network[network.iloc[:,2:].sum(axis=1)>0]
    influencedUsers = network.userID1.unique()
    results = []
    with multiprocessing.Pool() as p:
        results.extend(p.map(extractMessagesFromChunk, np.array_split(influencedUsers,multiprocessing.cpu_count())))
    all_messages = []
    for result in results:
        all_messages.extend(result)
    print(f'Messages contain : {len(all_messages)} lines')
    all_messages = pd.DataFrame(np.array(all_messages), columns = gEvents.columns)
    all_messages = all_messages.drop_duplicates().sort_values(by=["time","action"])
    all_messages.to_csv(f"{os.path.dirname(os.path.abspath(__file__))}/../init_data/MACM_Init_Messages.csv",index=False, date_format = "%Y-%m-%dT%H:%M:%SZ")
    print('Extract messages completed.')
############################################################################################


def NumerifyAndSubsetEvents(all_events):
    print("Numerifying events.")
    print("There are " + str(all_events.userID.unique().size) + " users. Considering all " + str((all_events.userID.unique().size ** 2) * (len(list(ACTION_MAP.keys())) **2 )) + " possible relationships")
    start = time.time()
    users_to_consider = all_events.groupby(["userID"]).apply(lambda x: x.set_index("time").resample("M").count().iloc[:,0].mean()>ACTIVITY_THRESHOLD[x.platform.iloc[0]])
    print(users_to_consider)
    users_to_consider = users_to_consider[users_to_consider==True].index.unique()
    all_events = all_events[all_events.userID.isin(users_to_consider)]
    print("There are " + str(all_events.userID.unique().size) + " users who are above activity threshold.")
    all_events, u, t = numerifyEvents(all_events)
    extractInfoIDProbDists(all_events, u)
    all_events = all_events[["userID","action","time"]].dropna()
    end = time.time()    
    print("Time taken to numerify event data: " + str(end-start) + " seconds.")
    return all_events, u, t

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", help="event file to be used to infer endo/exo-genous influence.")
    parser.add_argument("shocks_file", help="event file to be used to infer endo/exo-genous influence.")
    parser.add_argument("time_min", help="Start of training time.")
    parser.add_argument("time_max", help="End of training time.")
    parser.add_argument("-d", "--DeviceID", default=0, required=False, type=int, help="Device ID")
    args = parser.parse_args()
    events = pd.read_csv(args.event_file,parse_dates=["time"])
    events = events[['userID', 'nodeID', 'parentID', 'conversationID', 'time', 'action', 'platform', 'informationIDs']].copy()
    events["time"] = events.time.apply(lambda x: x.tz_localize(None))
    events = events[(events.time > dt.datetime.strptime(args.time_min, "%Y-%m-%dT%H:%M:%SZ")) & (events.time < dt.datetime.strptime(args.time_max, "%Y-%m-%dT%H:%M:%SZ"))]
    print(f"DeviceID : {args.DeviceID}")
    cuda.select_device(args.DeviceID)
    events, u, t = NumerifyAndSubsetEvents(events)
    network_te = extractEndogenousInfluence(events, u, t)
    extractMessages(args.event_file,network_te)
    del network_te
    shocks = pd.read_csv(args.shocks_file,parse_dates=["time"])
    shocks["time"] = shocks.time.apply(lambda x: x.tz_localize(None))
    shocks = shocks[(shocks.time > dt.datetime.strptime(args.time_min, "%Y-%m-%dT%H:%M:%SZ")) & (shocks.time < dt.datetime.strptime(args.time_max, "%Y-%m-%dT%H:%M:%SZ"))]
    extractExogenousInfluence(events, u, t, shocks)
    print('MACMInitialization completed execution.')

def _gpu_init_1d(n):
    """
    n = size of input data
    returns threads per block and blocks per thread tuple for 1d data for GPU init
    """
    threadsperblock = 128
    blockspergrid = int(math.ceil(n / threadsperblock))
    return (blockspergrid,threadsperblock)

def _gpu_init_2d(n,m):
    """
    n = 1st dim size of input data
    m = 2nd dim size of input data
    returns threads per block and blocks per thread tuple for 1d data for GPU init
    """
    threadsperblock = (16,16)
    blockspergrid_x = int(math.ceil(n/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(m/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return (blockspergrid,threadsperblock)

main()
