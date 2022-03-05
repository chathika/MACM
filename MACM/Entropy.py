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
This code containts cuda enabled transfer entropy, partial transfer entropy, entropy calculation functions 
to work with numba @cuda.jit annotations, required for the initialization of the Multi-Action Cascade Model.

"""

import math

from numba import cuda, jit
import numpy as np

@cuda.jit()
def calcH(events_matrix: np.array, H: np.array) -> None:
    """
    Numba device function. Each GPU thread will fill a position process an index cuda.grid(1) on 
    the events_matrix and fill a slot on the result array H.
    Shannon entropy calculated on all users. Event = 1, absence of event = 0. Units dits.

    :param events_matrix: 2D numpy array containing events/(absence of events) by all users over time.
    :param H: 1D numpy array to contain calculated entropy result of all users. 

    """
    userID = cuda.grid(1)
    if userID >= events_matrix.shape[0]:
        return
    userID = int(userID)
    p_1 = 0.0
    p_0 = 0.0
    for i in range(events_matrix.shape[1]):
        if events_matrix[userID, i] > 0:
            p_1 = p_1 + 1
        if events_matrix[userID, i] == 0:
            p_0 = p_0 + 1
    p_0 = float(p_0 / int(events_matrix.shape[1]))
    p_1 = float(p_1 / int(events_matrix.shape[1]))
    h = float(-1 * (p_1 * math.log(p_1) + p_0 * math.log(p_0)))
    H[userID] = h

@cuda.jit()
def calcPartialH(events_matrix: np.array, H: np.array) -> None:
    """
    Numba device function. Each GPU thread will fill a position process an index cuda.grid(1) on 
    the events_matrix and fill a slot on the result array H.
    Partial Shannon entropy calculated on all users. Only event = 1 considered. Units dits.

    :param events_matrix: 2D numpy array containing events/(absence of events) by all users over time.
    :param H: 1D numpy array to contain calculated entropy result of all users. 

    """
    userID = cuda.grid(1)
    if userID >= events_matrix.shape[0]:
        return
    userID = int(userID)
    p_1 = 0.0
    for i in range(events_matrix.shape[1]):
        if events_matrix[userID, i] > 0:
            p_1 = p_1 + 1
    p_1 = float(p_1 / int(events_matrix.shape[1]))
    h = float(-1 * (p_1 * math.log(p_1)))
    H[userID] = h


@cuda.jit()
def calcT(influencer_events_matrix: np.array, influencee_events_matrix: np.array, T: np.array) -> None:
    """
    Numba device function. Each GPU thread will fill a position process an index cuda.grid(1) on 
    the events_matrix and fill a slot on the result array T.
    Transfer entropy calculated on all users. Event = 1, absence of event = 0. Units dits.

    :param influencer_events_matrix: 2D numpy array containing numerified tensor representing events by all influencing users (axis=0) over time (axis=1).
    :param influencee_events_matrix: 2D numpy array containing numerified tensor representing events by all influenced users (axis=0) over time (axis=1).
    :param T: 1D numpy array to contain calculated transfer entropy. Flattened such that each slot is for (influencerID * num_influencees) + influenceeID. 

    """
    influencerID, influenceeID = cuda.grid(2)
    if influencerID >= influencer_events_matrix.shape[0] or influenceeID >= influencee_events_matrix.shape[0]:
        return
    influencerID = int(influencerID)
    influenceeID = int(influenceeID)
    # Calculate destination conditioned on past probabilities
    dest0_condition_past0_count = 0
    dest1_condition_past0_count = 0
    dest0_condition_past1_count = 0
    dest1_condition_past1_count = 0
    past0_count = 0
    past1_count = 0
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencee_events_matrix[influenceeID, i] == 0:
            past0_count = past0_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_past0_count = dest0_condition_past0_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_past0_count = dest1_condition_past0_count + 1
        if influencee_events_matrix[influenceeID, i] == 1:
            past1_count = past1_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_past1_count = dest0_condition_past1_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_past1_count = dest1_condition_past1_count + 1
    # Calculate destination _conditioned on past and source probabilities
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
        if influencer_events_matrix[influencerID, i] == 0 and influencee_events_matrix[influenceeID, i] == 0:
            source0_past0_count = source0_past0_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_source0_past0_count = dest0_condition_source0_past0_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source0_past0_count = dest1_condition_source0_past0_count + 1
        if influencer_events_matrix[influencerID, i] == 0 and influencee_events_matrix[influenceeID, i] > 0:
            source0_past1_count = source0_past1_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_source0_past1_count = dest0_condition_source0_past1_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source0_past1_count = dest1_condition_source0_past1_count + 1
        if influencer_events_matrix[influencerID, i] > 0 and influencee_events_matrix[influenceeID, i] == 0:
            source1_past0_count = source1_past0_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_source1_past0_count = dest0_condition_source1_past0_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source1_past0_count = dest1_condition_source1_past0_count + 1
        if influencer_events_matrix[influencerID, i] > 0 and influencee_events_matrix[influenceeID, i] > 0:
            source1_past1_count = source1_past1_count + 1
            if influencee_events_matrix[influenceeID, i+1] == 0:
                dest0_condition_source1_past1_count = dest0_condition_source1_past1_count + 1
            elif influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source1_past1_count = dest1_condition_source1_past1_count + 1
    TE = dest0_condition_source0_past0_count / (influencee_events_matrix.shape[1]-1) * math.log((dest0_condition_source0_past0_count / source0_past0_count) / (dest0_condition_past0_count / past0_count)) \
        + dest1_condition_source0_past0_count / (influencee_events_matrix.shape[1]-1) * math.log((dest1_condition_source0_past0_count / source0_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest0_condition_source0_past1_count / (influencee_events_matrix.shape[1]-1) * math.log((dest0_condition_source0_past1_count / source0_past1_count) / (dest0_condition_past1_count / past1_count)) \
        + dest1_condition_source0_past1_count / (influencee_events_matrix.shape[1]-1) * math.log((dest1_condition_source0_past1_count / source0_past1_count) / (dest1_condition_past1_count / past1_count)) \
        + dest0_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log((dest0_condition_source1_past0_count / source1_past0_count) / (dest0_condition_past0_count / past0_count)) \
        + dest1_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log((dest1_condition_source1_past0_count / source1_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest0_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log((dest0_condition_source1_past1_count / source1_past1_count) / (dest0_condition_past1_count / past1_count)) \
        + dest1_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log(
            (dest1_condition_source1_past1_count / source1_past1_count) / (dest1_condition_past1_count / past1_count))

    T[(influencerID*influencee_events_matrix.shape[0]) + influenceeID] = TE


@cuda.jit()
def calcPartialT(influencer_events_matrix: np.array, influencee_events_matrix: np.array, partialT: np.array) -> None:
    """
    Numba device function. Each GPU thread will fill a position process an index cuda.grid(1) on 
    the events_matrix and fill a slot on the result array T.
    Partial transfer entropy calculated on all users. Only event = 1 considered. Units dits.

    :param influencer_events_matrix: 2D numpy array containing numerified tensor representing events by all influencing users (axis=0) over time (axis=1).
    :param influencee_events_matrix: 2D numpy array containing numerified tensor representing events by all influenced users (axis=0) over time (axis=1).
    :param T: 1D numpy array to contain calculated partial transfer entropy. Flattened such that each slot is for (influencerID * num_influencees) + influenceeID. 

    """
    influencerID, influenceeID = cuda.grid(2)
    if influencerID >= influencer_events_matrix.shape[0] or influenceeID >= influencee_events_matrix.shape[0]:
        return
    influencerID = int(influencerID)
    influenceeID = int(influenceeID)
    # Calculate destination conditioned on past probabilities
    dest1_condition_past0_count = 0
    dest1_condition_past1_count = 0
    past0_count = 0
    past1_count = 0
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencee_events_matrix[influenceeID, i] == 0:
            past0_count = past0_count + 1
            if influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_past0_count = dest1_condition_past0_count + 1
        if influencee_events_matrix[influenceeID, i] == 1:
            past1_count = past1_count + 1
            if influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_past1_count = dest1_condition_past1_count + 1
    # Calculate destination _conditioned on past and source probabilities
    dest1_condition_source1_past0_count = 0
    dest1_condition_source1_past1_count = 0
    source1_past0_count = 0
    source1_past1_count = 0
    for i in range(influencee_events_matrix.shape[1]-1):
        if influencer_events_matrix[influencerID, i] > 0 and influencee_events_matrix[influenceeID, i] == 0:
            source1_past0_count = source1_past0_count + 1
            if influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source1_past0_count = dest1_condition_source1_past0_count + 1
        if influencer_events_matrix[influencerID, i] > 0 and influencee_events_matrix[influenceeID, i] > 0:
            source1_past1_count = source1_past1_count + 1
            if influencee_events_matrix[influenceeID, i+1] > 0:
                dest1_condition_source1_past1_count = dest1_condition_source1_past1_count + 1
    partial_t = dest1_condition_source1_past0_count / (influencee_events_matrix.shape[1]-1) * math.log((dest1_condition_source1_past0_count / source1_past0_count) / (dest1_condition_past0_count / past0_count)) \
        + dest1_condition_source1_past1_count / (influencee_events_matrix.shape[1]-1) * math.log(
            (dest1_condition_source1_past1_count / source1_past1_count) / (dest1_condition_past1_count / past1_count))

    partialT[(influencerID*influencee_events_matrix.shape[0]) +
             influenceeID] = partial_t

