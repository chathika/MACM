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
import os, sys
import numpy as np
from numba import cuda
import multiprocessing

start_time = input("Run start time (in \"%Y-%m-%dT%H:%M:%S%fZ\" format: ")
ticks_needed = input("Number of hours to run: ")
mmd_range = eval(input("MAX MEMORY DEPTH RANGE (in (min,max,step) format: "))
alpha_range = eval(input("ALPHA RANGE (in (min,max,step) format: "))
reps = int(eval(input("Number of repetitions per experiment: ")))

python_command = os.path.basename(sys.executable)

mmd_range = eval("range{0}".format(mmd_range))
alpha_range = eval("np.arange{0}".format(alpha_range))
total_runs = len(mmd_range) * len(alpha_range)
num_devices = len(cuda.list_devices())
i = 0


for rep in range(reps):
        for mmd in mmd_range:
                for alpha in alpha_range:
                        print("Running Rep={2}, MMD={0}, alpha={1}".format(mmd,alpha,rep))
                        os.system("{0} MACM2.py {1} {2} {3} {4} --quiet ".format(python_command, start_time, ticks_needed, mmd, alpha))