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
This file can be used to execute a single, manual run of the MACM
"""
import sys
# import os
# import glob
sys.path.insert(1,'./MACM/')
import MACM

START_TIME = '2017-02-01T00:00:00Z'
TICKS_TO_SIMULATE = 168
MAX_MEMORY_DEPTH = 10
MEMORY_DEPTH_FACTOR = 0.6
# DATA_FOLDER_PATH = os.path.join("..","InitData")
# print(DATA_FOLDER_PATH)
# print(glob.glob(os.path.join(DATA_FOLDER_PATH,'*')))
m = MACM.MACM(START_TIME, TICKS_TO_SIMULATE, MAX_MEMORY_DEPTH, MEMORY_DEPTH_FACTOR, QUIET_MODE = False, DEVICE_ID = 0, DUMP_AGENT_MEMORY = False)
m.run()