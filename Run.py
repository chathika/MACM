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

import argparse
from MACM import MACM
import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument("START_TIME", help="Start time of simulation.")#2018-01-01T00:00:00Z
parser.add_argument("TICKS_TO_SIMULATE", help="Number of hours to run simulation.")
parser.add_argument("MAX_MEMORY_DEPTH", help="Max memory depth parameter.")
parser.add_argument("MEMORY_DEPTH_FACTOR", help="Memory depth factor parameter.")
#parser.add_argument("--MAX_NUM_INFORMATION_IDS_PER_EVENT", type=int, default=1, required=False, help="Sets the maximum number of information IDs per event. Default is 1.")
parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Set for detailed output.")
parser.add_argument("--device-id", type=int, required=False, help="CUDA device id.")
parser.add_argument("-m", "--dump_agent_memory", action="store_true", default=False, help="Dump received information, actionable information, and attention span data. Considerably slows down model runs.")
args = parser.parse_args()



START_TIME = str(args.START_TIME)
TICKS_TO_SIMULATE = int(args.TICKS_TO_SIMULATE)
MAX_MEMORY_DEPTH = int(args.MAX_MEMORY_DEPTH)
MEMORY_DEPTH_FACTOR = float(args.MEMORY_DEPTH_FACTOR)


model = MACM.MACM(START_TIME, TICKS_TO_SIMULATE, MAX_MEMORY_DEPTH, MEMORY_DEPTH_FACTOR, QUIET_MODE = args.quiet, DEVICE_ID = args.device_id, DUMP_AGENT_MEMORY= args.dump_agent_memory, ENABLE_CONTENT_MUTATION = False, ENABLE_MODEL_P = True, ENABLE_MODEL_I = False)
model.run()