# MACM
The Multi-Action Cascade Model of Conversation

## Running MACMInitialization
`python MACM/MACMInitialization.py -h`
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


example (running on GPU device 1 ): 
`python MACM/MACMInitialization.py '/home/social-sim/SSDATA/CP4_Final/test_harness_1_macm_training_stance_no_dupes.csv' '/home/social-sim/MACMWorking/MACM/init_data/all_exogenous_shocks_cp4_scen1.csv' '2019-01-15T00:00:00Z' '2019-01-17T00:00:00Z' -d 1`


