# MACM
The Multi-Action Cascade Model of Conversation

## Running MACMInitialization

MACMInitialization file creates the data files requried for the MACM model to run and puts them in the init_data directory.

Following is the way to execute the `MACMInitialization.py` from the root directory:

```
usage: MACMInitialization.py [-h] [-d DEVICEID]
                             event_file shocks_file time_min time_max

positional arguments:
  event_file            event file to be used to infer endo/exo-genous
                        influence.
  shocks_file           event file to be used to infer endo/exo-genous
                        influence.
  time_min              Start of training time.
  time_max              End of training time.

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICEID, --DeviceID DEVICEID
                        Device ID
```

example (running on GPU device 1 ): 

```python MACM/MACMInitialization.py '/home/social-sim/SSDATA/CP4_Final/test_harness_1_macm_training_stance_no_dupes.csv' '/home/social-sim/MACMWorking/MACM/init_data/all_exogenous_shocks_cp4_scen1.csv' '2019-01-15T00:00:00Z' '2019-01-17T00:00:00Z' -d 1```


## Running the model
The model could be run using the Run.py or by creating your own running program which uses the class file MACM.

Following is the way to execute the `Run.py` from the root directory:

```
usage: Run.py [-h] [-q] [--device-id DEVICE_ID] [-m]
              START_TIME TICKS_TO_SIMULATE MAX_MEMORY_DEPTH
              MEMORY_DEPTH_FACTOR

positional arguments:
  START_TIME            Start time of simulation.
  TICKS_TO_SIMULATE     Number of hours to run simulation.
  MAX_MEMORY_DEPTH      Max memory depth parameter.
  MEMORY_DEPTH_FACTOR   Memory depth factor parameter.

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           Set for detailed output.
  --device-id DEVICE_ID
                        CUDA device id.
  -m, --dump_agent_memory
                        Dump received information, actionable information, and
                        attention span data. Considerably slows down model
                        runs.
```
Example (running on GPU device 0):

```python Run.py '2019-02-01T00:00:00Z' 3 10 0.8 --device-id 0```
