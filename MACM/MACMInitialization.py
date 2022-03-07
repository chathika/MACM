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


from typing import Tuple
import os
import math
import datetime as dt
import time
import sys
import pandas as pd
from numba import cuda, jit
import argparse
import numpy as np
from tqdm import tqdm
from Entropy import *
ACTIVITY_THRESHOLD = {
    'twitter': 10, 'youtube': 10, 'telegram': 10, 'github': 10}

EVENT_TO_ACTION_MAP = {
    "creation": ["CreateEvent", "tweet", "post", "Post", "video"],
    "contribution": ['IssueCommentEvent', 'PullRequestEvent',
                     'GollumEvent', 'PullRequestReviewCommentEvent', 'PushEvent',
                     'IssuesEvent', 'CommitCommentEvent', "DeleteEvent", "reply", "quote", "message", "comment", "Comment"],
    "sharing": ["ForkEvent", "WatchEvent", 'ReleaseEvent', 'MemberEvent', 'PublicEvent', "retweet"]
}

sys.path.append(os.path.dirname(__file__))


def getEventTypeIdx(event):
    """
    Helper function
    """
    for idx, name in enumerate(EVENT_TO_ACTION_MAP.keys()):
        if event in EVENT_TO_ACTION_MAP[name]:
            return idx


def ensurePlatformUniqueness(events):
    events = events.copy()
    events.loc[events.parentID.isna(
    ), "parentID"] = events.loc[events.parentID.isna(), "nodeID"]
    events.loc[events.conversationID.isna(
    ), "conversationID"] = events.loc[events.conversationID.isna(), "nodeID"]
    if("platform" in events.columns):
        events.loc[:, "userID"] = events.apply(
            lambda x: str(x.platform) + "_" + str(x.userID), axis=1)
        events.loc[:, "nodeID"] = events.apply(
            lambda x: str(x.platform) + "_" + str(x.nodeID), axis=1)
        events.loc[:, "parentID"] = events.apply(
            lambda x: str(x.platform) + "_" + str(x.parentID), axis=1)
        events.loc[:, "conversationID"] = events.apply(
            lambda x: str(x.platform) + "_" + str(x.conversationID), axis=1)
    return events


def numerifyEvents(events):
    """
    Sorts users in events in alphabetical order
    assigns numbering
    returns modified events and user mapping
    umapping: pandas dataframe of user names where names are columns (using this structure since it allows O(1) time access to user names)
    tmapping: same as umapping but for node ids (or 'targets')
    """
    events = ensurePlatformUniqueness(events)
    # print(events.head())
    # & events.parentID.isnull() & events.conversationID.isnull()]
    events = events[events.userID.notnull() & events.nodeID.notnull()]
    userlist = events.userID.unique()
    umapping = pd.Series(np.sort(list(set(userlist)))
                         ).reset_index().set_index(0).T
    events["userID"] = [umapping[x][0] for x in events["userID"]]

    targetlist = events.nodeID.append(events.parentID).append(
        events.conversationID).unique()
    tmapping = pd.Series(np.sort(list(set(targetlist)))
                         ).reset_index().set_index(0).T

    events["nodeID"] = [tmapping[x][0] for x in events["nodeID"]]
    #events["parentID"] = [ tmapping[x][0] for x in events["parentID"]]
    #events["conversationID"] = [ tmapping[x][0] for x in events["conversationID"]]
    events["action"] = events["action"].apply(lambda x: getEventTypeIdx(x))
    return (events, umapping, tmapping)


def _numerifyShocks(shocks_):
    """
    Sorts shocks in alphabetical order
    assigns numbering
    returns modified shocks mapping
    """
    shocks = shocks_.copy()
    shocklist = list(shocks.drop("time", axis=1).columns)
    smapping = pd.Series(np.sort(list(set(shocklist)))
                         ).reset_index().set_index(0).T
    shocks = shocks.set_index("time")
    shocks.columns = [smapping[col][0] for col in shocks.columns]
    shocks = shocks.reset_index()
    return (shocks, smapping)


@cuda.jit()
def _cuda_resample(compressed_events: np.array, resampled_events: np.array) -> None:
    userID = int(cuda.grid(1))
    if userID > compressed_events.shape[0]:
        return
    for i in range(compressed_events.shape[1]):
        time_delta = int(math.floor(compressed_events[userID, i]))
        if time_delta == -1:
            break
        resampled_events[userID,
                         time_delta] = resampled_events[userID, time_delta] + 1


def extract_endogenous_influence(events, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates entropy and partial entropy of users and social influence between users as
    transfer entropy and partial transfer entropy between user event timeseries.
    Start by assuming fully connected network
    make n*n matrix (n=number of users),  where each cell contains two activity timeseries
    rows = influencer
    cols = influencee

    :return: Tuple of 4 pd.DataFrame objects for user entropy, partial entropy, transfer entropy,
        and partial transfer entropy
    """
    numerified_events, u, t = _numerify_and_subset_events(events)
    H = pd.DataFrame()
    H_partial = pd.DataFrame()
    num_days = math.floor((numerified_events.time.max(
    ) - numerified_events.time.min()).total_seconds()/86400)
    chunk_size = 100
    for day_i in tqdm(range(0, max(chunk_size, num_days-chunk_size), chunk_size), desc='Progress processing endogenous social influence'):
        # Process data in chunks
        period_start = numerified_events.time.min() + dt.timedelta(days=day_i)
        period_end = numerified_events.time.min() + dt.timedelta(days=day_i + chunk_size)
        events = numerified_events[(numerified_events.time > period_start) & (
            numerified_events.time < period_end)].copy()
        if events.shape[0] == 0:
            break
        print("Generating event matrix.")
        time_max = events.time.max()
        time_min = events.time.min()
        # Ensure min and max timestamps, then resample and count
        # cudafy data
        events.loc[:, "time"] = events.time.apply(lambda x: int(
            (x - time_min).total_seconds()//3600))  # get time as hours float
        events = events[["userID", "action", "time"]
                        ].sort_values(["userID", "action"])
        max_events_per_user_action = events.groupby(["userID", "action"]).apply(
            lambda x: x.time).max()  # max user_action count
        events_matrix = np.zeros(
            (u.shape[1], len(list(EVENT_TO_ACTION_MAP.keys())), max_events_per_user_action))
        for action in tqdm(range(len(list(EVENT_TO_ACTION_MAP.keys()))), desc='Preparing events for cuda'):
            events_this_action = events[events["action"] == action]
            max_events_by_user_this_action = events_this_action.groupby(
                "userID").apply(lambda x: x.shape[0]).max()
            try:
                compressed_events = np.full(
                    (u.shape[1], max_events_by_user_this_action+1), -1.0)
            except:
                print("No events!")
                print(max_events_by_user_this_action)
            for userID in range(u.shape[1]):
                events_by_user_this_action = events_this_action[events_this_action["userID"] == userID]
                for i, event_time in enumerate(events_by_user_this_action.time):
                    compressed_events[userID, i] = event_time
            compressed_events = cuda.to_device(compressed_events)
            events_matrix_this_action = cuda.to_device(
                np.zeros((u.shape[1], max_events_per_user_action)))
            bpg, tpb = _gpu_init_1d(u.shape[1])
            _cuda_resample[bpg, tpb](
                compressed_events, events_matrix_this_action)
            events_matrix[:, action, :] = np.nan_to_num(
                events_matrix_this_action.copy_to_host())
            if verbose:
                print("Resampled and matrixified " +
                      str(list(EVENT_TO_ACTION_MAP.keys())[action]) + " events on GPU.")
        ###########################################################################################################
        # Calculate entropy per action
        H_chunk = np.zeros(
            (u.shape[1], len(list(EVENT_TO_ACTION_MAP.keys()))), dtype=np.float32)
        H_partial_chunk = np.zeros(
            (u.shape[1], len(list(EVENT_TO_ACTION_MAP.keys()))), dtype=np.float32)
        for action in tqdm(range(len(list(EVENT_TO_ACTION_MAP.keys()))), desc='Calculating Shannon entropy for users per action'):
            events_this_action = cuda.to_device(
                np.ascontiguousarray(events_matrix[:, action, :]))
            # Perform cuda calculations of H
            bpg, tpb = _gpu_init_1d(events_this_action.shape[0])
            H_chunk_action_device = cuda.to_device(
                np.zeros(events_this_action.shape[0], dtype=np.float32))
            calcH[bpg, tpb](events_this_action, H_chunk_action_device)
            H_chunk[:, action] = H_chunk_action_device.copy_to_host().tolist()
            # Perform cuda calculations of H
            H_partial_chunk_action_device = cuda.to_device(
                np.zeros(events_this_action.shape[0], dtype=np.float32))
            calcPartialH[bpg, tpb](
                events_this_action, H_partial_chunk_action_device)
            H_partial_chunk[:, action] = H_partial_chunk_action_device.copy_to_host(
            ).tolist()
        # Replace ordering placeholders with actual user IDs
        H_chunk = pd.DataFrame(H_chunk, columns=list(
            EVENT_TO_ACTION_MAP.keys())).fillna(0)
        H_chunk.index = H_chunk.index.set_names(['userID'])
        H_chunk = H_chunk.reset_index()
        H_chunk["userID"] = H_chunk["userID"].apply(lambda x: u.columns[x])
        H_chunk = H_chunk.set_index(["userID"])
        if verbose:
            print("Entropy calculations for chunk completed.")
        # Assimilate chunk measurements, Shannon entropy can be averaged over samples
        def take_mean(s1, s2): return (s1 + s2) / 2
        H = H.combine(H_chunk, func=take_mean, fill_value=0)
        del H_chunk
        H_partial_chunk = pd.DataFrame(H_partial_chunk, columns=list(
            EVENT_TO_ACTION_MAP.keys())).fillna(0)
        H_partial_chunk.index = H_partial_chunk.index.set_names(['userID'])
        H_partial_chunk = H_partial_chunk.reset_index()
        H_partial_chunk["userID"] = H_partial_chunk["userID"].apply(
            lambda x: u.columns[x])
        if verbose:
            print("Entropy calculations done.")
        H_partial_chunk = H_partial_chunk.set_index(["userID"])
        H_partial = H_partial.combine(
            H_partial_chunk, func=take_mean, fill_value=0)
        del H_partial_chunk
        ###########################################################################################################
        # Calculate Transfer Entropy per action->action relationship
        T = pd.DataFrame(index=pd.MultiIndex.from_product(
            [list(range(u.shape[1])), list(range(u.shape[1]))], names=["userID0", "userID1"]))
        T_partial = pd.DataFrame(index=pd.MultiIndex.from_product(
            [list(range(u.shape[1])), list(range(u.shape[1]))], names=["userID0", "userID1"]))
        T_chunk = pd.DataFrame()
        T_partial_chunk = pd.DataFrame()
        for influencer_action in tqdm(range(len(list(EVENT_TO_ACTION_MAP.keys()))), desc='Calculating transfer entropy for users per action-action relationship'):
            events_influencer_action = cuda.to_device(
                np.ascontiguousarray(events_matrix[:, influencer_action, :]))
            for influencee_action in range(len(list(EVENT_TO_ACTION_MAP.keys()))):
                events_influencee_action = cuda.to_device(
                    np.ascontiguousarray(events_matrix[:, influencee_action, :]))
                # Start cuda calculations of T
                bpg, tpb = _gpu_init_2d(
                    events_influencer_action.shape[0], events_influencee_action.shape[0])
                T_chunk_action_device = cuda.to_device(np.zeros(
                    (events_influencer_action.shape[0]*events_influencee_action.shape[0]), dtype=np.float32))
                calcT[bpg, tpb](events_influencer_action,
                                events_influencee_action, T_chunk_action_device)
                relationship_name = list(EVENT_TO_ACTION_MAP.keys())[
                    influencer_action] + "To" + list(EVENT_TO_ACTION_MAP.keys())[influencee_action]
                T_chunk_action = pd.DataFrame(T_chunk_action_device.copy_to_host().tolist(), columns=[relationship_name], index=pd.MultiIndex.from_product(
                    [list(range(u.shape[1])), list(range(u.shape[1]))], names=["userID0", "userID1"]))
                if T_chunk.empty:
                    T_chunk = T_chunk_action
                else:
                    T_chunk = T_chunk.join(T_chunk_action, how="outer")
                # Start cuda calculations of partial T
                T_partial_chunk_action_device = cuda.to_device(np.zeros(
                    (events_influencer_action.shape[0]*events_influencee_action.shape[0]), dtype=np.float32))
                calcPartialT[bpg, tpb](
                    events_influencer_action, events_influencee_action, T_partial_chunk_action_device)
                T_chunk_action = pd.DataFrame(T_partial_chunk_action_device.copy_to_host().tolist(), columns=[
                                              relationship_name], index=pd.MultiIndex.from_product([list(range(u.shape[1])), list(range(u.shape[1]))], names=["userID0", "userID1"]))
                if T_partial_chunk.empty:
                    T_partial_chunk = T_chunk_action
                else:
                    T_partial_chunk = T_partial_chunk.join(
                        T_chunk_action, how="outer")
                if verbose:
                    print("Transfer entropy for relationship " +
                          relationship_name + " done.")
        # Assimilate chunk measurements, transfer entropy can be averaged over samples
        T_chunk = T_chunk.reset_index().fillna(
            0.).set_index(["userID0", "userID1"])
        T = T.combine(T_chunk, func=take_mean, fill_value=0)
        del T_chunk
        T_partial_chunk = T_partial_chunk.reset_index().fillna(
            0.).set_index(["userID0", "userID1"])
        T_partial = T_partial.combine(
            T_partial_chunk, func=take_mean, fill_value=0)
        del T_partial_chunk

    # Replace user order placeholders with actual userIDs
    T = T.reset_index()
    T["userID0"] = T["userID0"].apply(lambda x: u.columns[x])
    T["userID1"] = T["userID1"].apply(lambda x: u.columns[x])
    T_partial = T_partial.reset_index()
    T_partial["userID0"] = T_partial["userID0"].apply(lambda x: u.columns[x])
    T_partial["userID1"] = T_partial["userID1"].apply(lambda x: u.columns[x])
    return (H, H_partial, T, T_partial)


def extract_exogenous_influence(events, shocks, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates entropy and partial entropy of exogenous shocks and influence of exogenous shocks as
    transfer entropy and partial transfer entropy from shock timeseries to user event timeseries, 
    per user action (creation, contribution, sharing).
    Start by assuming fully connected network
    make n*n matrix (n=number of users),  where each cell contains two activity timeseries
    rows = exogenous shock
    cols = influencee

    :return: Tuple of 4 pd.DataFrame objects for user entropy, partial entropy, transfer entropy,
        and partial transfer entropy
    """
    numerified_events, u, t = _numerify_and_subset_events(events)
    numerified_shocks, s = _numerifyShocks(shocks)
    H = pd.DataFrame()
    H_partial = pd.DataFrame()
    T = pd.DataFrame(index=pd.MultiIndex.from_product(
        [list(range(s.shape[1])), list(range(u.shape[1]))], names=["shock", "userID"]))
    T_partial = pd.DataFrame(index=pd.MultiIndex.from_product(
        [list(range(s.shape[1])), list(range(u.shape[1]))], names=["shock", "userID"]))
    num_days = math.floor((numerified_events.time.max(
    ) - numerified_events.time.min()).total_seconds()/86400)
    chunk_size = 100
    for day_i in tqdm(range(0, max(chunk_size, num_days-chunk_size), chunk_size), desc='Progress processing endogenous social influence'):
        period_start = numerified_events.time.min() + dt.timedelta(days=day_i)
        period_end = numerified_events.time.min() + dt.timedelta(days=day_i + chunk_size)
        events_chunk = numerified_events[(numerified_events.time > period_start) & (
            numerified_events.time < period_end)].copy()
        shocks_chunk = numerified_shocks[(numerified_shocks.time > period_start) & (
            numerified_shocks.time < period_end)].copy()
        if events_chunk.shape[0] == 0 or shocks_chunk.shape[0] == 0:
            break
        if verbose:
            print("Generating event matrix.")
        time_min = events_chunk.time.min()
        # Ensure min and max timestamps, then resample and count
        # cudafy data
        events_chunk.loc[:, "time"] = events_chunk.time.apply(
            lambda x: int((x - time_min).total_seconds()//3600))
        shocks_chunk.loc[:, "time"] = shocks_chunk.time.apply(
            lambda x: int((x - time_min).total_seconds()//3600))
        events_chunk = events_chunk[["userID", "action", "time"]
                                    ].sort_values(["userID", "action"])
        shocks_chunk = shocks_chunk.set_index("time")
        shocks_chunk = shocks_chunk[sorted(shocks_chunk.columns)]
        # Cudafy shocks
        max_events_per_user_action = events_chunk.groupby(
            ["userID", "action"]).apply(lambda x: x.time).max()
        max_times_shock_occurred = int(shocks_chunk.sum().max())
        compressed_shocks = np.full(
            (s.shape[1], max_times_shock_occurred+1), -1.0)
        for shockID in tqdm(range(s.shape[1]), desc='Preparing exogenous shocks for cuda'):
            times_this_shock_occurred = shocks_chunk[[shockID]].copy()
            times_this_shock_occurred = times_this_shock_occurred[times_this_shock_occurred[shockID] > 0].reset_index(
            )
            for i, shock_time in enumerate(times_this_shock_occurred.time):
                compressed_shocks[shockID, i] = shock_time
        compressed_shocks = cuda.to_device(compressed_shocks)
        shocks_matrix = cuda.to_device(
            np.zeros((s.shape[1], max_events_per_user_action)))
        bpg, tpb = _gpu_init_1d(s.shape[1])
        _cuda_resample[bpg, tpb](compressed_shocks, shocks_matrix)
        # Now do events
        events_matrix = np.zeros(
            (u.shape[1], len(list(EVENT_TO_ACTION_MAP.keys())), max_events_per_user_action))
        for action in tqdm(range(len(list(EVENT_TO_ACTION_MAP.keys()))), desc='Preparing events for cuda'):
            events_this_action = events_chunk[events_chunk["action"] == action]
            max_events_by_user_this_action = events_this_action.groupby(
                "userID").apply(lambda x: x.shape[0]).max()
            compressed_events = np.full(
                (u.shape[1], max_events_by_user_this_action+1), -1.0)
            for userID in range(u.shape[1]):
                events_by_user_this_action = events_this_action[events_this_action["userID"] == userID]
                for i, event_time in enumerate(events_by_user_this_action.time):
                    compressed_events[userID, i] = event_time
            compressed_events = cuda.to_device(compressed_events)
            events_matrix_this_action = cuda.to_device(
                np.zeros((u.shape[1], max_events_per_user_action)))
            bpg, tpb = _gpu_init_1d(u.shape[1])
            _cuda_resample[bpg, tpb](
                compressed_events, events_matrix_this_action)
            events_matrix[:, action, :] = np.nan_to_num(
                events_matrix_this_action.copy_to_host())
            if verbose:
                print("Resampled and matrixified " +
                      str(list(EVENT_TO_ACTION_MAP.keys())[action]) + " events on GPU.")
        ###########################################################################################################
        # Calculate Shannon entropy per shock
        # Perform cuda calculations of H
        bpg, tpb = _gpu_init_1d(shocks_matrix.shape[0])
        H_chunk_action_device = cuda.to_device(
            np.zeros(shocks_matrix.shape[0], dtype=np.float32))
        calcH[bpg, tpb](shocks_matrix, H_chunk_action_device)
        H_chunk = H_chunk_action_device.copy_to_host().tolist()
        # Perform cuda calculations of partial H
        H_partial_chunk_action_device = cuda.to_device(
            np.zeros(shocks_matrix.shape[0], dtype=np.float32))
        calcPartialH[bpg, tpb](shocks_matrix, H_partial_chunk_action_device)
        H_partial_chunk = H_partial_chunk_action_device.copy_to_host().tolist()
        #####
        H_chunk = pd.DataFrame(H_chunk, columns=["H"]).fillna(0)
        H_chunk.index = H_chunk.index.set_names(['shockID'])
        H_chunk = H_chunk.reset_index()
        H_chunk["shockID"] = H_chunk["shockID"].apply(lambda x: s.columns[x])
        H_chunk = H_chunk.set_index(["shockID"])
        def take_mean(s1, s2): return (s1 + s2) / 2
        H = H.combine(H_chunk, func=take_mean, fill_value=0)
        ######
        H_partial_chunk = pd.DataFrame(
            H_partial_chunk, columns=["H"]).fillna(0)
        H_partial_chunk.index = H_partial_chunk.index.set_names(['shockID'])
        H_partial_chunk = H_partial_chunk.reset_index()
        H_partial_chunk["shockID"] = H_partial_chunk["shockID"].apply(
            lambda x: s.columns[x])
        H_partial_chunk = H_partial_chunk.set_index(["shockID"])
        H_partial = H_partial.combine(
            H_partial_chunk, func=take_mean, fill_value=0)
        ###########################################################################################################
        # Calculate Transfer Entropy per action->action relationship
        T_chunk = pd.DataFrame()
        T_partial_chunk = pd.DataFrame()
        for influencee_action in tqdm(range(len(list(EVENT_TO_ACTION_MAP.keys()))), desc='Calculating transfer entropy for users per shock-action relationship'):
            events_influencee_action = cuda.to_device(
                np.ascontiguousarray(events_matrix[:, influencee_action, :]))
            # Start cuda calculations of T
            bpg, tpb = _gpu_init_2d(
                shocks_matrix.shape[0], events_influencee_action.shape[0])
            T_chunk_action_device = cuda.to_device(np.zeros(
                (shocks_matrix.shape[0]*events_influencee_action.shape[0]), dtype=np.float32))
            calcT[bpg, tpb](
                shocks_matrix, events_influencee_action, T_chunk_action_device)
            relationship_name = list(EVENT_TO_ACTION_MAP.keys())[
                influencee_action]
            T_chunk_action = pd.DataFrame(T_chunk_action_device.copy_to_host().tolist(), columns=[relationship_name], index=pd.MultiIndex.from_product(
                [list(range(s.shape[1])), list(range(u.shape[1]))], names=["shockID", "userID"]))
            if T_chunk.empty:
                T_chunk = T_chunk_action
            else:
                T_chunk = T_chunk.join(T_chunk_action, how="outer")
            # Start cuda calculations of partialT
            T_partial_chunk_action_device = cuda.to_device(np.zeros(
                (shocks_matrix.shape[0]*events_influencee_action.shape[0]), dtype=np.float32))
            calcPartialT[bpg, tpb](
                shocks_matrix, events_influencee_action, T_partial_chunk_action_device)
            T_partial_chunk_action = pd.DataFrame(T_partial_chunk_action_device.copy_to_host().tolist(), columns=[
                                                  relationship_name], index=pd.MultiIndex.from_product([list(range(s.shape[1])), list(range(u.shape[1]))], names=["shockID", "userID"]))
            if T_partial_chunk.empty:
                T_partial_chunk = T_partial_chunk_action
            else:
                T_partial_chunk = T_partial_chunk.join(
                    T_partial_chunk_action, how="outer")
            if verbose:
                print("Transfer entropy for relationship " +
                      relationship_name + " done.")
        # Assimilate chunk measurements, transfer entropy can be averaged over samples
        T_chunk = T_chunk.reset_index().fillna(
            0.).set_index(["shockID", "userID"])
        T = T.combine(T_chunk, func=take_mean, fill_value=0)
        T_partial_chunk = T_partial_chunk.reset_index().fillna(
            0.).set_index(["shockID", "userID"])
        T_partial = T_partial.combine(
            T_partial_chunk, func=take_mean, fill_value=0)

    T = T.reset_index()
    T["shockID"] = T["shockID"].apply(lambda x: s.columns[x])
    T["userID"] = T["userID"].apply(lambda x: u.columns[x])
    T_partial = T_partial.reset_index()
    T_partial["shockID"] = T_partial["shockID"].apply(lambda x: s.columns[x])
    T_partial["userID"] = T_partial["userID"].apply(lambda x: u.columns[x])
    return (H, H_partial, T, T_partial)


def extractMessages(events: pd.DataFrame, network: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Master function, splits the influenced user list and then asks workers to find their last n received messages.

    :param events: Social media events dataframe
    :param network: Endogenous transfer entropy dataframe, index 2 returned by extract_endogenous_influence
    :return: pd.DataFrame with latest messages received by users for MACM initialization

    """
    if verbose:
        print("Extracting Messages")
    gEvents = ensurePlatformUniqueness(events)
    if verbose:
        print('network:', network)
    influencerUsers = network.userID0.unique()
    all_messages = gEvents[gEvents.userID.isin(influencerUsers)]
    if verbose:
        print(f'Messages contain : {len(all_messages)} lines')
    all_messages = pd.DataFrame(
        np.array(all_messages), columns=gEvents.columns)
    all_messages = all_messages.drop_duplicates().sort_values(by=[
        "time", "action"])
    if verbose:
        print('Extract messages completed.')
    return all_messages


def _numerify_and_subset_events(all_events: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        print("Numerifying events.")
        print("There are " + str(all_events.userID.unique().size) + " users. Considering all " +
              str((all_events.userID.unique().size ** 2) * (len(list(EVENT_TO_ACTION_MAP.keys())) ** 2)) + " possible relationships")
    users_to_consider = all_events.groupby(["userID"]).apply(lambda x: x.set_index("time").resample(
        "M").count().iloc[:, 0].mean() > ACTIVITY_THRESHOLD[x.platform.iloc[0].lower()])
    users_to_consider = users_to_consider[users_to_consider == True].index.unique(
    )
    all_events = all_events[all_events.userID.isin(users_to_consider)]
    if verbose:
        print("There are " + str(all_events.userID.unique().size) +
              " users who are above activity threshold.")
    all_events, u, t = numerifyEvents(all_events)
    all_events = all_events[["userID", "action", "time"]].dropna()
    return all_events, u, t


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "event_file", help="event file to be used to infer endo/exo-genous influence.")
    parser.add_argument(
        "shocks_file", help="event file to be used to infer endo/exo-genous influence.")
    parser.add_argument("time_min", help="Start of training time.")
    parser.add_argument("time_max", help="End of training time.")
    parser.add_argument("-d", "--DeviceID", default=0,
                        required=False, type=int, help="Device ID")
    args = parser.parse_args()

    print(f"Working on Cuda device ID : {args.DeviceID}")
    cuda.select_device(args.DeviceID)

    events = pd.read_csv(args.event_file, parse_dates=["time"])[
        ['userID', 'nodeID', 'parentID', 'conversationID', 'time', 'action', 'platform', 'informationIDs']]
    events["time"] = events.time.apply(lambda x: x.tz_localize(None))
    events = events[(events.time > dt.datetime.strptime(args.time_min, "%Y-%m-%dT%H:%M:%SZ"))
                    & (events.time < dt.datetime.strptime(args.time_max, "%Y-%m-%dT%H:%M:%SZ"))]

    out_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'init_data')

    # Calculate endogenous social influence
    H, H_partial, T, T_partial = extract_endogenous_influence(events.copy())
    H.to_csv(os.path.join(out_dir, "MACM_Init_Endogenous_Entropy.csv"), index=True)
    H_partial.to_csv(os.path.join(
        out_dir, "MACM_Init_Endogenous_Partial_Entropy.csv"), index=True)
    T.to_csv(os.path.join(
        out_dir, "MACM_Init_Endogenous_Transfer_Entropy.csv"), index=True)
    T_partial.to_csv(os.path.join(
        out_dir, "MACM_Init_Endogenous_Partial_Transfer_Entropy.csv"), index=True)
    del H, H_partial, T_partial

    # Collect recent messages
    messages = extractMessages(events.copy(), T)
    del T

    # Calculate exogenous influence
    shocks = pd.read_csv(args.shocks_file, parse_dates=["time"])
    shocks["time"] = shocks.time.apply(lambda x: x.tz_localize(None))
    shocks = shocks[(shocks.time > dt.datetime.strptime(args.time_min, "%Y-%m-%dT%H:%M:%SZ"))
                    & (shocks.time < dt.datetime.strptime(args.time_max, "%Y-%m-%dT%H:%M:%SZ"))]
    H, H_partial, T, T_partial = extract_exogenous_influence(
        events.copy(), shocks.copy())
    H.to_csv(os.path.join(out_dir, "MACM_Init_Exogenous_Entropy.csv"), index=True)
    H_partial.to_csv(os.path.join(
        out_dir, "MACM_Init_Exogenous_Partial_Entropy.csv"), index=True)
    T.to_csv(os.path.join(
        out_dir, "MACM_Init_Exogenous_Transfer_Entropy.csv"), index=True)
    T_partial.to_csv(os.path.join(
        out_dir, "MACM_Init_Exogenous_Partial_Transfer_Entropy.csv"), index=True)
    print('MACMInitialization completed execution.')


def _gpu_init_1d(n: int) -> Tuple[int, int]:
    """
    Calculates threads per block and blocks per thread tuple for 1d data for GPU init.

    :param n: size of input data.
    :return: Threads per block and blocks per thread tuple for 1d data for GPU init.

    """
    threadsperblock = 128
    blockspergrid = int(math.ceil(n / threadsperblock))
    return (blockspergrid, threadsperblock)


def _gpu_init_2d(n: int, m: int) -> Tuple[int, int]:
    """
    Calculates threads per block and blocks per thread tuple for 1d data for GPU init.

    :param n: 1st dim size of input data.
    :param m: 2nd dim size of input data.
    :return: Threads per block and blocks per thread tuple for 1d data for GPU init.

    """
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(n/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(m/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return (blockspergrid, threadsperblock)


main()
