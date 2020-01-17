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
import numpy as np
import pandas as pd
import glob
import scipy.sparse as sps
from timeit import default_timer as timer
import numba
from numba import jit, prange, cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64, xoroshiro128p_normal_float32
import math
import datetime as dt
import time
import multiprocessing
from dateutil.tz import tzutc
from dateutil.parser import parse
from collections import OrderedDict
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("START_TIME", help="Start time of simulation.")#2018-01-01T00:00:00Z
parser.add_argument("TICKS_TO_SIMULATE", help="Number of hours to run simulation.")
parser.add_argument("MAX_MEMORY_DEPTH", help="Max memory depth parameter.")
parser.add_argument("MEMORY_DEPTH_FACTOR", help="Memory depth factor parameter.")
parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Set for detailed output.")
parser.add_argument("--device-id", type=int, required=False, help="CUDA device id.")
parser.add_argument("-m", "--dump_agent_memory", action="store_true", default=False, help="Dump received information, actionable information, and attention span data. Considerably slows down model runs.")
args = parser.parse_args()


MESSAGE_ITEM_COUNT = 6 #userID,action,nodeID,parentID,conversationID,rootID,informationIDs
START_TIME = dt.datetime.strptime(str(args.START_TIME), "%Y-%m-%dT%H:%M:%S%fZ")
TICKS_TO_SIMULATE = int(args.TICKS_TO_SIMULATE)
MAX_MEMORY_DEPTH = int(args.MAX_MEMORY_DEPTH)
MEMORY_DEPTH_FACTOR = float(args.MEMORY_DEPTH_FACTOR)
RECEIVED_INFORMATION_LIMIT=MAX_MEMORY_DEPTH + 50
event_types=OrderedDict({
        "creation": ["CreateEvent","tweet","post","Post"],
    "contribution": ['IssueCommentEvent', 'PullRequestEvent',
    'GollumEvent', 'PullRequestReviewCommentEvent', 'PushEvent', 
    'IssuesEvent', 'CommitCommentEvent',"DeleteEvent","reply","quote","message","comment","Comment"],
    "sharing": ["ForkEvent","WatchEvent", 'ReleaseEvent', 'MemberEvent', 'PublicEvent',"retweet"]
    })

def MACM_print(message):
    if (not args.quiet):
        print(message)

def getEventDictionary():
    
    return event_types


def getEventTypeIdx(action):    
    for idx,name in enumerate(getEventDictionary().keys()):
        if action in getEventDictionary()[name]:
            return idx    

def getEventTypes():
    return list(getEventDictionary().keys())


#@jit(parallel=True,nopython=False)
def propagate(edges,outgoing_messages,received_information):
    '''
    The propagate procedure of the MACM algorithm
    This function takes in the TE matrix and the outgoing messages of the agents as input
    @param: edges : [n,k] list of edges [row->col, k event_types]
    @param: outgoing_messages: [k,arbitrary] list of outgoing messages
    Both te and outgoing_messages assume that the rows and columns are ordered the same
    such that user ids are externally recoverable.
    @returns: received_information: ordered list of lists of received information of MACM agents
    '''
    new_received_information = np.copy(received_information)
    for message_idx in prange(len(outgoing_messages)):
        sender=outgoing_messages[message_idx][0]
        event_type=outgoing_messages[message_idx][1]
        receivers = edges[edges[:,0]==sender][:,1]
        receivers_new_memory = np.full((received_information.shape[1],MESSAGE_ITEM_COUNT),-1,dtype=np.float64)
        for receiver in receivers:
            receivers_new_memory[1:]=new_received_information[int(receiver),:-1]
            receivers_new_memory[0]=outgoing_messages[message_idx]
            new_received_information[int(receiver)]=receivers_new_memory
    return new_received_information



def Init():
    #Grab Endogenous Init Data
    MACM_print("Reading Endogenous Data")
    Data_Endo = {}
    Data_Endo["TE"] = pd.read_csv(glob.glob('InitData/*Endogenous_Partial_Transfer_Entropy*.csv')[0])
    Data_Endo["E"] = pd.read_csv(glob.glob('InitData/*Endogenous_Partial_Entropy*.csv')[0])
    df = Data_Endo["E"].rename(columns={"userID":"userID1"}).set_index("userID1")
    a = pd.DataFrame()
    for col1 in df.columns:
        for col2 in df.columns:
            a[col1 + "To" + col2] = df[col1]
    b = []
    for u in Data_Endo["TE"].userID1.unique():
        c = a.copy().reset_index()
        c["userID0"] = u
        b.extend(c.values.tolist())
    cols=["userID1"]
    cols.extend(list(a.columns))
    cols.extend(["userID0"])
    Data_Endo["E"] = pd.DataFrame(b,columns=cols)
    del a,b,c
    #Grab Exogenous Init Data
    MACM_print("Reading Exogenous Data")
    Data_Exo = {}
    Data_Exo["TE"] = pd.read_csv(glob.glob('InitData/*Exogenous_Partial_Transfer_Entropy*.csv')[0])
    Data_Exo["E"] = pd.read_csv(glob.glob('InitData/*Endogenous_Partial_Entropy*.csv')[0])#pd.read_csv(glob.glob('InitData/*Exogenous_Partial_Entropy*.csv')[0])
    df = Data_Exo["E"].set_index("userID")
    a = pd.DataFrame()
    for col1 in getEventTypes():
        a[col1] = df[col1]
    b = []
    for shock in Data_Exo["TE"].shockID.unique():
        c = a.copy().reset_index()
        c["shockID"] = shock
        b.extend(c.values.tolist())
    cols=["userID"]
    cols.extend(list(a.columns))
    cols.extend(["shockID"])
    Data_Exo["E"] = pd.DataFrame(b,columns=cols)
    del b,c
    #Ensure user consistency across dataframes
    MACM_print("Joining datasets to ensure data for all users")
    joined_endo = Data_Endo["TE"].set_index(["userID0","userID1"]).sort_index().join(Data_Endo["E"].set_index(["userID0","userID1"]).sort_index(),how="outer",lsuffix="TE",rsuffix="E").fillna(0).sort_index().reset_index()
    joined_endo = joined_endo[joined_endo.userID0 != joined_endo.userID1]
    joined_exo = Data_Exo["TE"].set_index(["shockID","userID"]).sort_index().join(Data_Exo["E"].set_index(["shockID","userID"]).sort_index(),how="outer",lsuffix="TE",rsuffix="E").fillna(0).sort_index().reset_index()
    MACM_print("Numerifying user and shock ids")
    userlist= joined_endo.reset_index().userID1.append(joined_endo.reset_index().userID0).append(joined_exo.reset_index().userID).unique().tolist()
    joined_exo=joined_exo[joined_exo["userID"].isin(userlist)]    
    #Numerify the user ids
    umapping = pd.Series(np.sort(list(set(userlist)))).reset_index().set_index(0).T
    shocklist = joined_exo.reset_index().shockID.unique()
    smapping = pd.Series(np.sort(list(set(shocklist)))).reset_index().set_index(0).T
    joined_endo["userID0"] = [ umapping[x][0] for x in joined_endo["userID0"]]
    joined_endo["userID1"] = [ umapping[x][0] for x in joined_endo["userID1"]]
    joined_exo["shockID"] = [ smapping[x][0] for x in joined_exo["shockID"]]
    joined_exo["userID"] = [ umapping[x][0] for x in joined_exo["userID"]]
    #reindex
    joined_endo = joined_endo.sort_values(["userID0","userID1"]).set_index(["userID0","userID1"])
    joined_exo = joined_exo.sort_values(["shockID","userID"]).set_index(["shockID","userID"])
    #### Convert to Matrices
    MACM_print("Setting Up Endogenous Influence")
    relationshipsTE=[]
    relationshipsE=[]
    for i in getEventTypes():
        for j in getEventTypes():
            relationshipsTE.append(str(i) + "To" + str(j) + "TE")
            relationshipsE.append(str(i) + "To" + str(j) + "E")
    ####    
    non_zero_joined_endo = joined_endo[joined_endo[relationshipsTE].sum(axis=1) > 0]
    df = (non_zero_joined_endo[relationshipsTE].fillna(0) / (non_zero_joined_endo[relationshipsE].fillna(0).values)).fillna(0)
    Data_Endo["q"] = np.empty((df.shape[0],len(getEventTypes()),len(getEventTypes())))
    for ida in range(len(getEventTypes())):
        for idb in range(len(getEventTypes())):
            Data_Endo["q"][:,ida,idb] = df.iloc[:,(len(getEventTypes()) * ida) + idb].values
    Data_Endo["edges"]=df.reset_index()[["userID0","userID1"]].values
    Data_Endo["Influencer_Index"]=np.full(umapping.size,-1)
    u0 = 0
    for idx in range(Data_Endo["edges"].shape[0]):    
        if Data_Endo["edges"][idx,0] >= u0:
            Data_Endo["Influencer_Index"][Data_Endo["edges"][idx,0]] = idx
            u0 = Data_Endo["edges"][idx,0] + 1
     #######
    MACM_print("Setting Up Exogenous Influence")
    MACM_print(joined_exo.head())
    non_zero_joined_exo = joined_exo[joined_exo[[e + "TE" for e in getEventTypes()]].sum(axis=1) > 0]
    df = (non_zero_joined_exo[ [e + "TE" for e in getEventTypes()] ].fillna(0) / (non_zero_joined_exo[[e + "E" for e in getEventTypes()]].fillna(0).abs().values)).fillna(0)
    Data_Exo["p"] = df.values
    Data_Exo["edges"]=df.reset_index()[["shockID","userID"]].values
    Data_Exo["Shock_Index"]=np.full(smapping.size,-1)
    s0 = 0
    for idx in range(Data_Exo["edges"].shape[0]):    
        if Data_Exo["edges"][idx,0] >= s0:
            Data_Exo["Shock_Index"][Data_Exo["edges"][idx,0]] = idx
            s0 = Data_Exo["edges"][idx,0] + 1
    ########
    #Calculate I: I = In
    MACM_print("Setting Up Internal Effect")
    df = joined_endo[relationshipsTE].reset_index().groupby("userID1").apply(lambda x: x.sum()).drop(["userID0","userID1"],axis=1).rename(index={"userID1":"userID"}).copy()
    df1 = pd.DataFrame()
    for eventType in getEventTypes():
        influencingCols = []
        for col in df.columns:
            if ("To" + eventType) in col:
                influencingCols.append(col)
        df1 = df1.join(df[influencingCols].sum(axis=1).rename(eventType + "EndoInf"),how="outer")
    df2 = joined_exo[[e + "TE" for e in getEventTypes()]].reset_index().groupby("userID").apply(lambda x: x.sum()).drop(["shockID","userID"],axis=1).copy()
    df2.columns = [ col[:-2] + "ExoInf" for col in df2.columns]
    df1 = df1.join(df2)
    df = joined_endo[relationshipsE].reset_index().groupby("userID0").apply(lambda x: x.mean()).drop(["userID0","userID1"],axis=1).rename(index={"userID1":"userID"}).copy()
    for eventType in getEventTypes():
        influencingCols = []
        for col in df.columns:
            if ("To" + eventType) in col:
                influencingCols.append(col)
        df1 = df1.join(df[influencingCols].sum(axis=1).rename(eventType + "IntInf"),how="outer")
    df1 = df1.fillna(0)
    df = pd.DataFrame()
    for eventType in getEventTypes():
        df = df.join( ((df1[(eventType + "IntInf")] - df1[(eventType + "EndoInf")] - df1[(eventType + "ExoInf")])/ df1[(eventType + "IntInf")].values).rename(eventType),how="outer")
    prob = pd.read_csv(glob.glob('InitData/*Endogenous_Hourly_Activity_Probability*.csv')[0])
    prob["userID"] = [ umapping[x][0] for x in prob["userID"]]
    prob = prob.set_index("userID")
    Data_Endo["I"] = (prob.fillna(0) * df.fillna(0).values).values
    MACM_print(Data_Endo["I"])
    #Set up the messages matrix
    MACM_print("Initializing messages")
    MessagesToPropagate=[]#np.empty((0))
    Data_Msg=pd.read_csv(glob.glob("InitData/*Messages*.csv")[0],parse_dates=["time"]).set_index("time").tz_localize(None).reset_index().sort_values("time",ascending=True)
    #Data_Msg=Data_Msg[Data_Msg.time > START_TIME-dt.timedelta(days=1)]
    Data_Msg=Data_Msg[Data_Msg["userID"].isin(userlist)]
    Data_Msg["userID"]=[ umapping[x][0] for x in Data_Msg["userID"]] 
    #construct target conversation mapping and numerify nodeID parentID and conversationIDs
    targetlist=Data_Msg["nodeID"].append(Data_Msg["parentID"]).append(Data_Msg["conversationID"])
    tmapping = np.sort(list(set(targetlist)))
    Data_Msg["nodeID"] = Data_Msg.apply(lambda x: np.nonzero(tmapping==x.nodeID)[0][0],axis=1)
    Data_Msg["parentID"] = Data_Msg.apply(lambda x: np.nonzero(tmapping==x.parentID)[0][0],axis=1)
    Data_Msg["conversationID"] = Data_Msg.apply(lambda x: np.nonzero(tmapping==x.conversationID)[0][0],axis=1)
    #construct information ID mapping and numerify informationIDs
    informationIDslist=Data_Msg["informationIDs"].unique()
    imapping = pd.Series(np.sort(list(set(informationIDslist)))).reset_index().set_index(0).T
    Data_Msg["informationIDs"] = [ imapping[str(x)][0] for x in Data_Msg["informationIDs"]]
    #Data_Msg = Data_Msg[["userID","nodeID","parentID","conversationID","action","time"]].dropna()
    ReceivedInformation=np.full((umapping.size,RECEIVED_INFORMATION_LIMIT,MESSAGE_ITEM_COUNT),-1,dtype=np.float64)
    MessagesToPropagate = []
    for idx,msg in Data_Msg.iterrows():
        MessagesToPropagate.append(np.array([msg.userID,int(getEventTypeIdx(msg.action)),int(msg.nodeID),int(msg.parentID),int(msg.conversationID),int(msg.informationIDs)]))
    ReceivedInformation = propagate(Data_Endo["edges"],MessagesToPropagate,ReceivedInformation)
    #Ensure they don't start overloaded
    for userID, user_ris in enumerate(ReceivedInformation):
        riID = random.randint(0,MAX_MEMORY_DEPTH)
        while riID < RECEIVED_INFORMATION_LIMIT:
            for msgitm in range(MESSAGE_ITEM_COUNT):
                ReceivedInformation[userID,riID,msgitm] = -1
            riID += 1
    MACM_print("Initializing exogenous shocks to simulation")
    shocks=pd.read_csv(glob.glob("InitData/*shocks*.csv")[0],parse_dates=["time"])
    shocks=shocks[shocks.time > START_TIME].sort_values("time",ascending=False).set_index("time").resample("60S").sum().fillna(0)
    Data_Exo["shocks"]=np.full((shocks.shape[0],smapping.size),-1,dtype=np.int32)
    for idx,shock in enumerate(smapping):
        if shock in shocks.columns:
            Data_Exo["shocks"][:,idx] = shocks[shock].values
        else: 
            Data_Exo["shocks"][:,idx]=np.full(shocks.shape[0],0,dtype=np.int32)
    return (Data_Endo,Data_Exo,ReceivedInformation,umapping,tmapping,smapping,imapping)


et=len(getEventTypes())
creation_idx = list(getEventDictionary().keys()).index("creation")
@cuda.jit()
def step(rng_states,inf_idx_Qs,edges_Qs,Qs,inf_idx_Ps,edges_Ps,Ps,p_by_action,messages,outgoing_messages,shocks,Is,uniq):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    influencee_id = int(cuda.grid(1))
    if influencee_id >= messages.shape[0]:
        # Quit if (x, y) is outside of valid C boundary
        return
    event_number = 0.0
    for outgoing_message_idx in range(outgoing_messages.shape[1]):
        for message_item in range(outgoing_messages.shape[2]):
            outgoing_messages[influencee_id,outgoing_message_idx,message_item]=-1
    #process shocks in memory
    for message_idx in range(messages.shape[1]):
        most_recent_influencer_id = int(messages[influencee_id,message_idx,0])
        most_recent_influencer_action = int(messages[influencee_id,message_idx,1])
        if most_recent_influencer_id >= 0 and most_recent_influencer_action == -1:
            #This is a shock message from before
            shock_ID = int(messages[influencee_id,message_idx,0])
            edge_idx_P = inf_idx_Ps[shock_ID]
            #if edge_idx_P is -1 then 
            while (edge_idx_P >=0) and (edge_idx_P < edges_Ps.shape[0]) and (edges_Ps[edge_idx_P,0] == shock_ID):
                if edges_Ps[edge_idx_P][0] == shock_ID and edges_Ps[edge_idx_P][1] == influencee_id:
                    break
                edge_idx_P = edge_idx_P + 1
            for possible_influencee_action in range(et):
                p_by_action[influencee_id,possible_influencee_action] = p_by_action[influencee_id,possible_influencee_action] * (1- Ps[edge_idx_P,possible_influencee_action])
    for possible_influencee_action in range(et):
        p_by_action[influencee_id,possible_influencee_action] = 1 - p_by_action[influencee_id,possible_influencee_action]
    outgoing_message_idx = 0
    for message_idx in range(messages.shape[1]):
        most_recent_influencer_id = int(messages[influencee_id,message_idx,0])
        most_recent_influencer_action = int(messages[influencee_id,message_idx,1])
        if most_recent_influencer_id >= 0 and most_recent_influencer_action >= 0:
            #IDs established, get the Q edge index
            edge_idx_Q = inf_idx_Qs[most_recent_influencer_id]
            while (edge_idx_Q >= 0) and (edge_idx_Q < edges_Qs.shape[0]) and (edges_Qs[edge_idx_Q,0] == most_recent_influencer_id):
                if edges_Qs[edge_idx_Q][0] == most_recent_influencer_id and edges_Qs[edge_idx_Q][1] == influencee_id:
                    break
                edge_idx_Q = edge_idx_Q + 1
            #Use edge index to get q value
            possible_influencee_action=int(0)
            influencee_action_taken=int(-1)
            influencee_action_prob=0
            while possible_influencee_action < et:
                message_qt = Qs[edge_idx_Q,most_recent_influencer_action,possible_influencee_action]
                message_pt = p_by_action[influencee_id,possible_influencee_action]
                rnd =  xoroshiro128p_uniform_float64(rng_states, influencee_id)
                prob = message_qt(message_qt + message_pt - (message_qt * message_pt))
                if rnd < prob:
                    if influencee_action_prob < prob:
                        influencee_action_taken=int(possible_influencee_action)
                        influencee_action_prob=prob    
                possible_influencee_action = int(possible_influencee_action + 1)
            if influencee_action_prob > 0 and influencee_action_taken >= 0:
                #construct outgoing message
                outgoing_messages[influencee_id,outgoing_message_idx,0] = influencee_id
                outgoing_messages[influencee_id,outgoing_message_idx,1] = influencee_action_taken
                outgoing_messages[influencee_id,outgoing_message_idx,2] = int(uniq[0] + influencee_id) + (event_number / RECEIVED_INFORMATION_LIMIT)
                event_number += 1        
                if np.int32(influencee_action_taken) != np.int32(creation_idx):
                    outgoing_messages[influencee_id,outgoing_message_idx,3] = int(messages[influencee_id,message_idx,2]) #parentID
                    outgoing_messages[influencee_id,outgoing_message_idx,4] = int(messages[influencee_id,message_idx,4]) 
                else: 
                    outgoing_messages[influencee_id,outgoing_message_idx,4] = int(outgoing_messages[influencee_id,outgoing_message_idx,2])#conversationID ...is nodeID if action is creation
                    outgoing_messages[influencee_id,outgoing_message_idx,3] = int(outgoing_messages[influencee_id,outgoing_message_idx,2]) #parentID ...is nodeID if action is creation
                outgoing_messages[influencee_id,outgoing_message_idx,5] = int(messages[influencee_id,message_idx,5])
                outgoing_message_idx=outgoing_message_idx+1
                #Remove message from RI
                #This is a simple pop, but doing it manually because numba/cuda
                #First, shift the stack up
                ai_idx = message_idx
                while ai_idx < MAX_MEMORY_DEPTH - 1:
                    for message_item in range(MESSAGE_ITEM_COUNT):
                        #shift message
                        messages[influencee_id,ai_idx,int(message_item)] = messages[influencee_id,ai_idx+1,int(message_item)]
                    ai_idx = ai_idx + 1
                #Finally, push in the empty message
                for message_item in range(MESSAGE_ITEM_COUNT):
                        messages[influencee_id,MAX_MEMORY_DEPTH - 1,int(message_item)] = -1 
    
    '''for possible_action, I in enumerate(Is[influencee_id,:]):
        rnd =  xoroshiro128p_uniform_float64(rng_states, influencee_id)
        if rnd < I:
            #construct outgoing message
            outgoing_messages[influencee_id,outgoing_message_idx,0] = influencee_id
            outgoing_messages[influencee_id,outgoing_message_idx,1] = possible_action
            outgoing_messages[influencee_id,outgoing_message_idx,2] = int(uniq[0] + influencee_id) + (event_number / RECEIVED_INFORMATION_LIMIT)
            event_number += 1
            outgoing_messages[influencee_id,outgoing_message_idx,3] = -1 #parentID unknown
            outgoing_messages[influencee_id,outgoing_message_idx,4] = -1 #conversationID unknown
            outgoing_messages[influencee_id,outgoing_message_idx,5] = -1 #int(messages[influencee_id,message_idx,5])
            outgoing_message_idx=outgoing_message_idx+1
    '''

@cuda.jit()
def propagate_gpu(inf_idx,edges,outgoing_messages,received_information):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sender_id = int(cuda.grid(1))
    if sender_id >= received_information.shape[0]:
        # Quit if (x, y) is outside of valid C boundary
        return
    #Repeat for all valid outgoing_messages
    outgoing_message_idx = 0
    while outgoing_messages[int(sender_id),outgoing_message_idx,0] > -1:
        #Find this influencees receivers and send them the message
        edge_idx = int(inf_idx[sender_id])
        while (edge_idx >= 0) and (edge_idx < edges.shape[0]) and (edges[edge_idx,0] == sender_id):
            #This is an outgoing edge of this influencee
            #get receivers ID
            receiver_id = int(edges[edge_idx,1])
            if (receiver_id != sender_id) and (receiver_id != outgoing_messages[int(sender_id),outgoing_message_idx,0]) and (outgoing_messages[int(sender_id),outgoing_message_idx,0] >=0 ) and (receiver_id >=0):
                #Add this message to it's received_information stack
                #This is a simple pop and push, but doing it manually because numba/cuda
                #First, shift the stack down
                i = RECEIVED_INFORMATION_LIMIT-1
                while i  > 0:
                    for message_item in range(MESSAGE_ITEM_COUNT):
                        received_information[receiver_id,i,int(message_item)] = received_information[receiver_id,i-1,int(message_item)]
                    i = i - 1
                #Finally, push in the new message
                for message_item in range(MESSAGE_ITEM_COUNT):
                        received_information[receiver_id,0,int(message_item)] = outgoing_messages[int(sender_id),outgoing_message_idx,int(message_item)]
            edge_idx = edge_idx + 1
        #move to next outgoing message
        outgoing_message_idx=outgoing_message_idx+1


@cuda.jit()
def recompute_memory_gpu(rng_states,received_information, actionable_information, cmd, shocks, shuffled_ri_idxs):
    influencee_id = int(cuda.grid(1))
    if influencee_id >= received_information.shape[0]:
        return
    current_memory_depth_influencee = cmd[influencee_id]
    #find the length of the actionable information stack
    for current_ai_length in range(MAX_MEMORY_DEPTH):
        mean_info = 0
        for idx, info in enumerate(actionable_information[influencee_id,current_ai_length]):
            mean_info = mean_info + info
        if mean_info / (idx + 1) == -1:
            break
    #find the length of the received information stack
    for current_ri_length in range(RECEIVED_INFORMATION_LIMIT):
        mean_info = 0
        for idx, info in enumerate(received_information[influencee_id,current_ri_length]):
            mean_info = mean_info + info
        if mean_info / (idx + 1) == -1:
            break
    #Also, add outgoing messages for each active shock
    for shock_ID, shock in enumerate(shocks):
        if shock > 0 and current_ri_length < RECEIVED_INFORMATION_LIMIT:
            #construct outgoing message with action -1
            received_information[influencee_id,int(current_ri_length),0] = shock_ID
            received_information[influencee_id,int(current_ri_length),1] = -1
            received_information[influencee_id,int(current_ri_length),2] = -1
            received_information[influencee_id,int(current_ri_length),3] = -1
            received_information[influencee_id,int(current_ri_length),4] = -1
            received_information[influencee_id,int(current_ri_length),5] = -1
            current_ri_length=current_ri_length+1
    for i in range(current_ri_length):
        shuffled_ri_idxs[i] = i
    for i in range(current_ri_length,0,-1):
        rnd_idx =  int(xoroshiro128p_uniform_float64(rng_states, influencee_id) * i)
        tmp = shuffled_ri_idxs[i]
        shuffled_ri_idxs[i] = shuffled_ri_idxs[rnd_idx]
        shuffled_ri_idxs[rnd_idx] = tmp
    #Excess messages is the number of things you have to respond to already plus the new things you have to respond to minus the number of things you can respond to
    current_overload_influencee = current_ai_length + current_ri_length - MAX_MEMORY_DEPTH
    #Of course, overload cannot be negative
    current_overload_influencee = current_overload_influencee if current_overload_influencee > 0 else 0
    #Reduction in current memory depth is the overload to the power of the memory depth factor
    current_memory_depth_influencee = current_memory_depth_influencee - (current_overload_influencee ** MEMORY_DEPTH_FACTOR)
    current_memory_depth_influencee = current_memory_depth_influencee if current_memory_depth_influencee >= 0 else 0
    cmd[influencee_id] = current_memory_depth_influencee
    #Transfer received information to actionable information
    ri_idx = current_ri_length - 1
    while ri_idx >= 0:
        #This is a simple pop and push, but doing it manually because numba/cuda
        #First, pull the stack down
        ai_idx = MAX_MEMORY_DEPTH-1
        while ai_idx  > 0:
            for message_item in range(MESSAGE_ITEM_COUNT):
                #Remove actionable information past the current memory depth 
                if ai_idx > current_memory_depth_influencee - 1:
                    actionable_information[influencee_id,ai_idx,int(message_item)] = -1
                else:
                    #else shift message
                    actionable_information[influencee_id,ai_idx,int(message_item)] = actionable_information[influencee_id,ai_idx-1,int(message_item)]
            ai_idx = ai_idx - 1
        #Finally, push in the new message
        for message_item in range(MESSAGE_ITEM_COUNT):
                actionable_information[influencee_id,0,int(message_item)] = received_information[influencee_id,shuffled_ri_idxs[ri_idx],int(message_item)]
        ri_idx = ri_idx - 1
    #Clear received information
    for ri_idx in range(int(current_ri_length)):
        for message_item in range(MESSAGE_ITEM_COUNT):
            received_information[influencee_id,ri_idx,message_item]=-1



Data_Endo,Data_Exo,Received_Information,umapping,tmapping,smapping,imapping = Init()
#print(smapping)
###### Copy the arrays to the device ######
#Endogenous influence arrays
Q_global_mem = cuda.to_device(Data_Endo["q"])
Endo_Inf_Idx_global_mem = cuda.to_device(Data_Endo["Influencer_Index"])
Endo_Edges_global_mem = cuda.to_device(Data_Endo["edges"])
#Exogenous influence arrays)
P_global_mem = cuda.to_device(Data_Exo["p"])
Exo_Inf_Idx_global_mem = cuda.to_device(Data_Exo["Shock_Index"])
Exo_Edges_global_mem = cuda.to_device(Data_Exo["edges"])
shocks_tmp_counter = cuda.to_device(np.full((umapping.size,RECEIVED_INFORMATION_LIMIT),0,dtype=np.float64))
#Internal
I_global_mem = cuda.to_device(Data_Endo["I"])
#Message arrays
#Received_Information=np.full((umapping.size,RECEIVED_INFORMATION_LIMIT,MESSAGE_ITEM_COUNT),-1,dtype=np.float64)
ri_global_mem = cuda.to_device(Received_Information)
# Allocate memory on the device for the result
outgoing_messages=np.full((umapping.size,RECEIVED_INFORMATION_LIMIT,MESSAGE_ITEM_COUNT),-1,dtype=np.float64)
om_global_mem = cuda.to_device(outgoing_messages)
actionable_information=np.full((Received_Information.shape[0],MAX_MEMORY_DEPTH,MESSAGE_ITEM_COUNT),-1,dtype=np.float64)
ai_global_mem = cuda.to_device(actionable_information)
current_memory_depths=np.full(Received_Information.shape[0],MAX_MEMORY_DEPTH,dtype=np.float64)
cmd_global_mem = cuda.to_device(current_memory_depths)
# Configure the blocks
TPB=16
threadsperblock = TPB
blockspergrid = int(math.ceil(Received_Information.shape[0] / threadsperblock))
rng_states = create_xoroshiro128p_states(blockspergrid * threadsperblock, seed=1)

all_events=[]
cmd_t=[]
recI_t=[]
actI_t=[]
diag_time_start = time.time()
s = 0
ticks = TICKS_TO_SIMULATE
while s < ticks:
    if args.dump_agent_memory:
        all_ri_this_tick = ri_global_mem.copy_to_host()
        for influencee_id, ri_influencee in enumerate(all_ri_this_tick):
            for ri in ri_influencee:
                if ri[0] > -1:
                    info = [s,influencee_id]
                    for ri_item in ri:
                        info.append(ri_item)
                    recI_t.append(info)
    #recalculate memory state
    exo_shocks_global_mem = cuda.to_device(Data_Exo["shocks"][s].copy())
    recompute_memory_gpu[blockspergrid,threadsperblock](rng_states,ri_global_mem,ai_global_mem,cmd_global_mem,exo_shocks_global_mem)
    if args.dump_agent_memory:
        all_ai_this_tick = ai_global_mem.copy_to_host()
        for influencee_id, ai_influencee in enumerate(all_ai_this_tick):
            for ai in ai_influencee:
                if ai[0] > -1:
                    info = [s,influencee_id]
                    for ai_item in ai:
                        info.append(ai_item)
                    actI_t.append(info)
    #Act on received messages
    #Calculate new endogenous influence
    p_by_action_global_mem = cuda.to_device(np.full((umapping.size,et),1,dtype=np.float64))
    uniq_global_mem = cuda.to_device(np.array([int(s << math.ceil(math.log(Received_Information.shape[0],2)))]))
    step[blockspergrid,threadsperblock](rng_states,Endo_Inf_Idx_global_mem,Endo_Edges_global_mem,Q_global_mem,Exo_Inf_Idx_global_mem,Exo_Edges_global_mem,P_global_mem,p_by_action_global_mem,ai_global_mem,om_global_mem,exo_shocks_global_mem,I_global_mem,uniq_global_mem)
    #Diagnostics
    events=om_global_mem.copy_to_host()
    for events_by_influencee in events:
        for event in events_by_influencee:
            if event[1] != -1:
                message = [s]
                for message_item_idx in range(MESSAGE_ITEM_COUNT):
                    message.append(event[message_item_idx])
                MACM_print(message)
                all_events.append(message)
    if args.dump_agent_memory:
        cmds = cmd_global_mem.copy_to_host()
        for idx, cmd in enumerate(cmds):
            cmd_t.append([s,int(idx),math.floor(cmd)])
    #Propagate
    propagate_gpu[blockspergrid,threadsperblock](Endo_Inf_Idx_global_mem,Endo_Edges_global_mem,om_global_mem,ri_global_mem)  
    MACM_print(START_TIME + dt.timedelta(hours = s))
    s = s + 1

diag_time_stop = time.time()
print("Took : " + str(diag_time_stop-diag_time_start) + " seconds to run " + str((START_TIME - START_TIME + dt.timedelta(hours = s)).days) + " days")


all_events=pd.DataFrame(all_events,columns = ["time","userID","action","nodeID","parentID","conversationID","informationIDs"])
all_events["time"]=all_events.iloc[:,0].apply(lambda x: START_TIME + dt.timedelta(hours = x))
all_events["userID"]=[ umapping.columns[int(x)] for x in all_events.iloc[:,1]]
all_events["action"]=all_events.iloc[:,2].apply(lambda x: getEventTypes()[int(x)])
#all_events["nodeID"]=all_events.iloc[:,3].apply(lambda x: tmapping[int(x)] if int(x) + 1 < len(tmapping) else x)
all_events["parentID"]=all_events.iloc[:,4].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
all_events["conversationID"]=all_events.iloc[:,5].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
all_events["informationIDs"]=[ imapping.columns[int(x)] for x in all_events.iloc[:,6]]
identifier = str(dt.datetime.now())
all_events.to_csv("output/MACM_MMD{0}_Alpha{1}_{2}.csv".format(MAX_MEMORY_DEPTH,MEMORY_DEPTH_FACTOR,identifier),index=False)

if args.dump_agent_memory:
    recI_t=pd.DataFrame(recI_t,columns = ["time","influenceeID","influencerID","action","nodeID","parentID","conversationID","informationIDs"])
    print(recI_t)
    recI_t["time"]=recI_t.iloc[:,0].apply(lambda x: START_TIME + dt.timedelta(hours = x))
    recI_t["influenceeID"]=[ umapping.columns[int(x)] if x > -1 else x for x in recI_t.iloc[:,1]]
    recI_t["influencerID"]=[ umapping.columns[int(x)] if x > -1 else x for x in recI_t.iloc[:,2]]
    recI_t["action"]=recI_t.iloc[:,3].apply(lambda x: getEventTypes()[int(x)] if x > -1 else x)
    #recI_t["nodeID"]=recI_t.iloc[:,4].apply(lambda x: tmapping[int(x)] if int(x) + 1 < len(tmapping) else x)
    #recI_t["parentID"]=recI_t.iloc[:,5].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
    #recI_t["conversationID"]=recI_t.iloc[:,6].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
    #recI_t["informationIDs"]=[ imapping.columns[int(x)] for x in recI_t.iloc[:,7]]
    actI_t=pd.DataFrame(actI_t,columns = ["time","influenceeID","influencerID","action","nodeID","parentID","conversationID","informationIDs"])
    print(actI_t)
    actI_t["time"]=actI_t.iloc[:,0].apply(lambda x: START_TIME + dt.timedelta(hours = x))
    actI_t["influenceeID"]=[ umapping.columns[int(x)] if x > -1 else x for x in actI_t.iloc[:,1]]
    actI_t["influencerID"]=[ umapping.columns[int(x)] if x > -1 else x for x in actI_t.iloc[:,2]]
    actI_t["action"]=actI_t.iloc[:,3].apply(lambda x: getEventTypes()[int(x)] if x > -1 else x)
    #actI_t["nodeID"]=actI_t.iloc[:,4].apply(lambda x: tmapping[int(x)] if int(x) + 1 < len(tmapping) else x)
    #actI_t["parentID"]=actI_t.iloc[:,5].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
    #actI_t["conversationID"]=actI_t.iloc[:,6].apply(lambda x: tmapping[int(x)] if (int(x) >= 0 ) and (int(x) + 1 < len(tmapping)) else x)
    #actI_t["informationIDs"]=[ imapping.columns[int(x)] for x in actI_t.iloc[:,7]]
    cmd_t=pd.DataFrame(cmd_t,columns=["time","userID","CMD"])
    cmd_t["time"]=cmd_t.iloc[:,0].apply(lambda x: START_TIME + dt.timedelta(hours = x))
    cmd_t["userID"]=[ umapping.columns[int(x)] if int(x) > -1 else int(x) for x in cmd_t.iloc[:,1]]
    
    
if args.dump_agent_memory:
    recI_t.to_csv("output/MACM_RI_MMD{0}_Alpha{1}_{2}.csv".format(MAX_MEMORY_DEPTH,MEMORY_DEPTH_FACTOR,identifier),index=False)
    actI_t.to_csv("output/MACM_AI_MMD{0}_Alpha{1}_{2}.csv".format(MAX_MEMORY_DEPTH,MEMORY_DEPTH_FACTOR,identifier),index=False)
    cmd_t.to_csv("output/MACM_CMD_MMD{0}_Alpha{1}_{2}.csv".format(MAX_MEMORY_DEPTH,MEMORY_DEPTH_FACTOR,identifier),index=False)
"""
all_events.groupby('userID').apply(lambda x: x.count()).mean()
all_events.groupby('userID').apply(lambda x: x.count()).median()
all_events.groupby('userID').apply(lambda x: x.count()).max()
all_events.groupby('userID').apply(lambda x: x.count()).min()

all_events.userID.unique().size / Data_Endo["TE"].shape[0]

############################################################################################
#### Process Ground Truth
import matplotlib.pyplot as plt
df=all_events[all_events.time > (START_TIME + dt.timedelta(days=0))]
df=df.set_index("time").groupby("action").apply(lambda x: x.resample("D").count()).drop("action",axis=1).reset_index()
pd.pivot_table(df,index="time",columns="action",values="userID").plot()

gt = pd.read_csv("cp2sc2_training_plus_ic.csv",parse_dates=["time"]).set_index("time").tz_localize(None).reset_index()
gt = gt[(gt.time > (START_TIME + dt.timedelta(days=0))) & (gt.time <= START_TIME + dt.timedelta(hours = s))]
gt.action = gt.action.apply(lambda x:  getEventTypes()[getEventTypeIdx(x)])
dfgt=gt.set_index("time").groupby("action").apply(lambda x: x.resample("D").count()).drop("action",axis=1).reset_index()
pd.pivot_table(dfgt,index="time",columns="action",values="userID").plot()
plt.show()

pd.Series(cmd_global_mem.copy_to_host()).hist(bins=MAX_MEMORY_DEPTH)
plt.show()

cmd_t = pd.DataFrame(cmd_t)
cmd_t.columns = ["time","cmd"]
cmd_t["time"]=cmd_t.time.apply(lambda x: START_TIME + dt.timedelta(hours = x))
df = cmd_t.set_index("time").resample("H").mean()
df.plot()
plt.show()"""

#make a replay
'''import random
total_hours = (gt.time.max() - gt.time.min()).total_seconds()/3600
user_prob = gt.groupby(["userID","action"]).apply(lambda x: x.count()/total_hours).iloc[:,0].rename("count").reset_index()
replay_events = []
for tick in range(ticks):
    for user in user_prob.userID.unique():
        probs = user_prob[user_prob.userID==user]
        roulette_wheel = np.array([])
        for potential_action in probs.action.unique():
            roulette_wheel = np.append(roulette_wheel,probs[probs.action==potential_action]["count"].iloc[0])
        roulette_wheel = [x / np.sum(roulette_wheel) for x in roulette_wheel]
        roulette_wheel = np.cumsum(roulette_wheel)
        rand = random.random()
        for action_idx, peice in enumerate(roulette_wheel):
            if rand < peice :
                break
        action = probs.action.unique().tolist()[action_idx]
        if random.random() < probs[probs.action == action]["count"].iloc[0]:
            replay_events.append([START_TIME + dt.timedelta(hours = tick), user, action, "","",""])
    print(START_TIME + dt.timedelta(hours = tick))

'''
'''replay_events = pd.DataFrame(replay_events,columns = ["time","userID","action","nodeID","parentID","conversationID"])


fig,ax=plt.subplots(1)

df=replay_events.set_index("time").groupby("action").apply(lambda x: x.resample("D").count()).drop("action",axis=1).reset_index()
plt1=pd.pivot_table(df,index="time",columns="action",values="userID")df=all_events[all_events.time > (START_TIME + dt.timedelta(days=0))]
plt2=df.set_index("time").groupby("action").apply(lambda x: x.resample("D").count()).drop("action",axis=1).reset_index()
plt3=pd.pivot_table(dfgt.reset_index(),index = "time",columns ="action",values ="userID")["time","userID"]
plt1.join(plt2,lsuffix = " Recent Freq ", rsuffix = " MACM").join(plt3,rsuffix=" Ground Truth").plot()
plt.show()

#df= df.set_index("time").iloc[:,:2]
#sq_err_replay = ((pd.pivot_table(dfgt.reset_index(),index = "time",columns ="action",values ="userID").diff() - pd.pivot_table(df.reset_index(),index = "time",columns ="action",values ="userID").diff()) ** 2)


df=all_events[all_events.time > (START_TIME + dt.timedelta(days=0))]
df=df.set_index("time").groupby("action").apply(lambda x: x.resample("D").count()).drop("action",axis=1).reset_index()

#sq_err_macm = ((pd.pivot_table(dfgt.reset_index(),index = "time",columns ="action",values ="userID").diff() - pd.pivot_table(df.reset_index(),index = "time",columns ="action",values ="userID")).diff() ** 2)

'''
'''
plt1 = replay_events[["time","userID","action"]]#.set_index(["time","userID"])
plt1["model"] = "Recent Freq"
plt0 = gt[["time","userID","action"]]#.set_index(["time","userID"])
plt0["model"] = "Ground Truth"
plt3 = all_events[["time","userID","action"]]#.set_index(["time","userID"])
plt3["model"] = "MACM"

plt.rcParams["xtick.labelsize"]=20
plt.rcParams["ytick.labelsize"]=20
fig, axs = plt.subplots(nrows = 3,sharex=True)
#fig.add_subplot(111, frameon=False)

all_comp=plt0.append(plt1).append(plt3)
all_comp = all_comp[all_comp.model.isin(["MACM"])]
all_comp = pd.pivot_table(all_comp.reset_index(),index="time",columns=["model","action"],aggfunc="count").userID.resample("D").sum()
all_comp.plot(ax=axs[0])
axs[0].set_title("MACM",fontsize=22)
axs[0].get_legend().remove()

all_comp=plt0.append(plt1).append(plt3)
all_comp = all_comp[all_comp.model.isin(["Ground Truth"])]
all_comp = pd.pivot_table(all_comp.reset_index(),index="time",columns=["model","action"],aggfunc="count").userID.resample("D").sum()
all_comp.plot(ax=axs[1])
axs[1].set_title("Ground Truth",fontsize=22)
axs[1].legend(loc="center left", bbox_to_anchor=(1,0.5),labels=["Sharing","Creation","Contribution"])

all_comp=plt0.append(plt1).append(plt3)
all_comp = all_comp[all_comp.model.isin(["Recent Freq"])]
all_comp = pd.pivot_table(all_comp.reset_index(),index="time",columns=["model","action"],aggfunc="count").userID.resample("D").sum()
all_comp.plot(ax=axs[2])
axs[2].set_title("Recent Freq",fontsize=22)
axs[2].get_legend().remove()
#for ax in axs:
#    ax.tick_params(which = "major',labelsize = 28)
#plt.tick_params(axis="both", which ="major", labelsize=22)
fig.text(0.04,0.5,  "Daily Event Count", va="center",rotation= 90,fontsize=22)
#plt.ylabel(, fontsize = 32, labelpad = 10)
plt.show()
'''