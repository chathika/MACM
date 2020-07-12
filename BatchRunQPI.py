import multiprocessing
import argparse
from MACM import MACM
import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument("START_TIME", help="Start time of simulation.")
parser.add_argument("TICKS_TO_SIMULATE", help="Number of hours to run simulation.")
parser.add_argument("MAX_MEMORY_DEPTH", help="Max memory depth parameter.")
parser.add_argument("MEMORY_DEPTH_FACTOR", help="Memory depth factor parameter.")
parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Set for detailed output.")
parser.add_argument("-m", "--dump_agent_memory", action="store_true", default=False, help="Dump received information, actionable information, and attention span data. Considerably slows down model runs.")
args = parser.parse_args()

START_TIME = str(args.START_TIME)
TICKS_TO_SIMULATE = int(args.TICKS_TO_SIMULATE)
MAX_MEMORY_DEPTH = int(args.MAX_MEMORY_DEPTH)
MEMORY_DEPTH_FACTOR = float(args.MEMORY_DEPTH_FACTOR)

numReps = 2

def CreateModelObject(modelType, deviceID):
    Param_ENABLE_MODEL_P = False
    Param_ENABLE_MODEL_I = False
    Param_ENABLE_CONTENT_MUTATION = False
    modelType = modelType.upper()
    if 'P' in modelType:
        Param_ENABLE_MODEL_P = True
    if 'I' in modelType:
        Param_ENABLE_MODEL_I = True
    if 'CA' in modelType:
        Param_ENABLE_CONTENT_MUTATION = True
    print([modelType, deviceID, Param_ENABLE_CONTENT_MUTATION, Param_ENABLE_MODEL_P, Param_ENABLE_MODEL_I])
    ret_model = MACM.MACM(START_TIME, TICKS_TO_SIMULATE, MAX_MEMORY_DEPTH, MEMORY_DEPTH_FACTOR, QUIET_MODE = args.quiet, 
                   DEVICE_ID = deviceID, DUMP_AGENT_MEMORY= args.dump_agent_memory, 
                   ENABLE_CONTENT_MUTATION = Param_ENABLE_CONTENT_MUTATION, ENABLE_MODEL_P = Param_ENABLE_MODEL_P, ENABLE_MODEL_I = Param_ENABLE_MODEL_I)
    return ret_model

def CreateAndRunModelReplicas(modelType, deviceID):
    model = CreateModelObject(modelType, deviceID)
    for i in range(numReps):
        model.run()

def main():
    paramSet = [('Q_CA',0,),   ('Q',1),
                ('QI_CA',2),  ('QI',3),
                ('QP_CA',4),  ('QP',5),
                ('PQI_CA',6), ('PQI',7)]
    workers = []
    for ps in paramSet:
        temp_process = multiprocessing.Process(target=CreateAndRunModelReplicas, args=ps)
        workers.append(temp_process)
    
    for w in workers:
        w.start()

main()