import pandas as pd
import random
import os
import numpy as np
import datetime
import json
import re
import configparser
import sys
import copy
import multiprocessing
#from collections import OrderedDict

# profiling
#import cProfile

# Following global variables are used by functions below. Each multiprocess process will have its own copy of this variable.
nextNodeID = -13.0
nextUserID = -21.0

class Parameters:
	def __init__(self, in_ParamFilePath):
		self.ParamsFilePath = in_ParamFilePath
		self.ReadParams()
	
	def ReadParams(self):
		config = configparser.ConfigParser()
		config.read(self.ParamsFilePath)
		# DEFAULT Main Runtype Option
		multi_run = config['DEFAULT']['multi_run']
		# DEFAULT Common Options
		ScenarioNo = config['DEFAULT']['ScenarioNo']
		Sim_StartTime = config['DEFAULT']['Sim_StartTime']
		Sim_EndTime = config['DEFAULT']['Sim_EndTime']
		CurrentChallenge = config['DEFAULT']['CurrentChallenge']
		CurrentSprint = config['DEFAULT']['CurrentSprint']
		CommitSha = config['DEFAULT']['CommitSha']
		output_directory_path = config['DEFAULT']['output_directory_path']
		nodelist_file = config['DEFAULT']['nodelist_file']
		ModelConfigIdentifier = config['DEFAULT']['ModelConfigIdentifier']
		IdentifierStr = config['DEFAULT']['IdentifierStr']
		GroupDesc = config['DEFAULT']['GroupDesc']
		RunBy = config['DEFAULT']['RunBy']
		RunNumber = config['DEFAULT']['RunNumber']
		mynote = config['DEFAULT']['mynote']
		# Multiple Files Optioins
		mr_csv_folder = config['MultipleFiles']['mr_csv_folder']
		# Single File Options
		file_csv_eventlog = config['SingleFiles']['file_csv_eventlog']
		file_csv_S3Location = config['SingleFiles']['file_csv_S3Location']
		Model_MemoryDepth = config['SingleFiles']['Model_MemoryDepth']
		Model_OverloadFactor = config['SingleFiles']['Model_OverloadFactor']
		RunGroup = config['SingleFiles']['RunGroup']
		fileTypeKey = 'MultipleFiles' if eval(multi_run) else 'SingleFiles'
		#print(fileTypeKey)
		#for key in config[fileTypeKey]:
		#	print('Key: ' + key + '\n\t--> Value: ' + config[fileTypeKey][key] + '\n\t--> Eval: ' + str(eval(config[fileTypeKey][key])) + '\n\t--> EvalType: ' + str(type(eval(config[fileTypeKey][key]))))
		# -- Assign parameters --
		# Runtype Option
		self.multi_run = eval(multi_run)
		# Common Options
		self.ScenarioNo = eval(ScenarioNo)
		self.Sim_StartTime = eval(Sim_StartTime)
		self.Sim_EndTime = eval(Sim_EndTime)
		self.CurrentChallenge = CurrentChallenge
		self.CurrentSprint = eval(CurrentSprint)
		self.CommitSha = eval(CommitSha)
		self.output_directory_path = eval(output_directory_path)
		self.nodelist_file = eval(nodelist_file)
		self.IdentifierStr = eval(IdentifierStr)
		self.ModelConfigIdentifier = eval(ModelConfigIdentifier)
		self.GroupDesc = eval(GroupDesc)
		self.RunBy = eval(RunBy)
		self.RunNumber = eval(RunNumber)
		self.mynote = eval(mynote)
		# Multiple Files Optioins
		self.mr_csv_folder = eval(mr_csv_folder)
		# Single File Options
		self.file_csv_eventlog = eval(file_csv_eventlog)
		self.file_csv_S3Location = eval(file_csv_S3Location)
		self.Model_MemoryDepth = eval(Model_MemoryDepth)
		self.Model_OverloadFactor = eval(Model_OverloadFactor)
		self.RunGroup = eval(RunGroup)
		self.mpLogQueue = None # must avoid deepcopying this when it is initialized!
		# -- Done --
	
	def deepcopy(self):
		return copy.deepcopy(self)
	
	def __str__(self):
		return	'Parameters:\n\tMulti Run: {}'	\
				'\n\tScenario Number: {}'		\
				'\n\tSim_StartTime: {}'			\
				'\n\tSim_EndTime : {}'			\
				'\n\tCurrentChallenge : {}'		\
				'\n\tCurrentSprint : {}'		\
				'\n\tCommitSha : {}'			\
				'\n\toutput_directory_path : {}'\
				'\n\tnodelist_file : {}'		\
				'\n\tIdentifierStr : {}'		\
				'\n\tModelConfigIdentifier : {}'\
				'\n\tGroupDesc : {}'			\
				'\n\tRunBy : {}'				\
				'\n\tRunNumber : {}'			\
				'\n\tmynote : {}'				\
				'\n\tmr_csv_folder : {}'		\
				'\n\tfile_csv_eventlog : {}'	\
				'\n\tfile_csv_S3Location : {}'	\
				'\n\tModel_MemoryDepth : {}'	\
				'\n\tModel_OverloadFactor : {}'	\
				'\n\tRunGroup : {}'.format(
										self.multi_run,self.ScenarioNo,self.Sim_StartTime,self.Sim_EndTime , self.CurrentChallenge,
										self.CurrentSprint , self.CommitSha , self.output_directory_path , 
										self.nodelist_file , self.IdentifierStr, self.ModelConfigIdentifier, self.GroupDesc , self.RunBy , self.RunNumber , 
										self.mynote , self.mr_csv_folder , self.file_csv_eventlog , 
										self.file_csv_S3Location , self.Model_MemoryDepth , self.Model_OverloadFactor , 
										self.RunGroup 
										)
	
	def __repr__(self):
		return self.__str__()
	

def MultiRun(in_Params):
	print("--Begining of Script--")
	# Following global variable is a multithreading protected queue
	mainLogQueue = multiprocessing.Manager().Queue()
	if not in_Params.multi_run:
		in_Params.mpLogQueue = mainLogQueue
		Run(in_Params)
	else:
		fileParamsList = []
		for f in os.listdir(in_Params.mr_csv_folder):
			if 'MACM' == f[:4] and '.csv' == f[-4:]:
				print('-- Processing File :' + f + ' --')
				fileParams = in_Params.deepcopy()
				fileParams.multi_run = False
				fileParams.file_csv_eventlog = os.path.join(in_Params.mr_csv_folder,f)
				temp = re.search('MMD[0-9]*',f)
				fileParams.Model_MemoryDepth = f[temp.span()[0] + 3 : temp.span()[1]]
				temp = re.search('Alpha[0-9.]*',f)
				fileParams.Model_OverloadFactor = float(f[temp.span()[0] + 5 : temp.span()[1]])
				fileParams.RunGroup = 'cp{}_sp{}_MACMGPU{}'.format(str(fileParams.CurrentChallenge), str(fileParams.CurrentSprint), str(fileParams.ModelConfigIdentifier))
				fileParams.mpLogQueue = mainLogQueue
				fileParamsList.append(fileParams)
		with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
			p.map(Run, fileParamsList)
	
	if mainLogQueue != None and not mainLogQueue.empty():
		logfilename = in_Params.output_directory_path + "/" + "log-" + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S-%f") + ".csv"
		loglist = []
		while not mainLogQueue.empty():
			loglist.append(mainLogQueue.get_nowait())
		dfMainLog = pd.DataFrame(loglist,columns=['filename', 'nodeTime_msc', 'nodeUserID_n1c', 'actionType_n1c', 'actionType_uec', 'nodeID_n1c', 'parentID_n1c', 'rootID_n1c', 'informationID_uidc', 'informationID_n1c', 'informationID_vmic', 'informationID_mic','informationID_uiidc','informationID_uiids', 'platform_n1c', 'platform_mpc', 'eventlogjson','mynote'])
		dfMainLog.to_csv(logfilename, index_label='FileIndex')
		print("\nThe Script Log file at : " + logfilename)
	print("--End of Script--")

def Run(in_Params):
	global nextNodeID
	global nextUserID
	#global dfMainLog

	thislog = {}
	thislog['filename'] = in_Params.file_csv_eventlog

	if not os.path.exists(in_Params.output_directory_path):
		print ("The Output directory path does not exist!")
		exit()

	# The File Regex:
	# let validName = /\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2}-\d{3,6}-[a-zA-Z0-9_]{1,30}-\d+-[TRGYM][WDHTGP]-[sS][cC][012]\.json/;
	thisDT = datetime.datetime.now()
	file_name_prefix = thisDT.strftime("%m-%d-%Y-%H-%M-%S-%f") + "-" + str(in_Params.RunGroup) + "-" + str(in_Params.RunNumber)
	file_eventLog = in_Params.output_directory_path + "/" + file_name_prefix + "-MP-Sc" + str(in_Params.ScenarioNo) + ".json"
	file_metadata = in_Params.output_directory_path + "/" + file_name_prefix + "-xMetadata.json"

	print('\treading event log file...')
	print('\t\tFile Name: ' + in_Params.file_csv_eventlog)
	originalData = pd.read_csv(in_Params.file_csv_eventlog,dtype={'userID':str,'conversationID':str,'parentID':str,'nodeID':str, 'informationIDs':str},na_values=str, parse_dates=['time'])
	print('\trenaming columns...')
	originalData.rename(columns={'userID':'nodeUserID','action':'actionType','time':'nodeTime','conversationID':'rootID','informationIDs':'informationID'},inplace=True)
	print('\t\tShape is : ' + str(originalData.shape))
	print('\tcalculating next UserID and NodeID values...')
	nextUserID = np.nansum([originalData['nodeUserID'].apply(pd.to_numeric,errors='coerce').max(), 537])
	print('\t\tNext User ID : ' + str(nextUserID))
	nextNodeID = np.nansum([originalData['nodeID'].apply(pd.to_numeric,errors='coerce').max() , 213])
	print('\t\tNext Node ID : ' + str(nextNodeID))

	# -----------------------------------MAPPING OF EVENTS---------------------
	platformToEventTypeToEvent = {
		'twitter':{
					"creation": ["tweet"],
					"contribution": ["reply","quote"],
					"sharing": ["retweet"] 
				},
		'github':{
					"creation": ["CreateEvent"],
					"contribution": ['IssueCommentEvent', 'PullRequestEvent', 'PullRequestReviewCommentEvent', 'PushEvent','IssuesEvent', 'CommitCommentEvent',"DeleteEvent"],
					"sharing": ["ForkEvent","WatchEvent"]
				},
		'reddit':{
					"creation": ["post"],
					"contribution": ["comment"],
					"sharing": ['comment']
				},
		'telegram':{
					"creation": ['message'],
					"contribution": ["message"],
					"sharing": ['message']
				},
		'youtube':{
					"creation": ['video'],
					"contribution": ["comment"],
					"sharing": ['comment']
				}
	}

	def MapEventUniformlyRandom(record):
		temp = platformToEventTypeToEvent[record['platform']][record['actionType']]
		if len(temp) < 1:
			return 'UNKNOWN-EVENT'
		return random.choice(temp)

	def MapEventProbablyRandom(record):
		temp = platformToEventTypeToEvent[record['platform']][record['actionType']]
		if len(temp) < 1:
			return 'UNKNOWN-EVENT'
		if record['platform'] == 'twitter' and record['actionType'] == 'contribution':
			return 'reply' if random.random() < 0.6698681489736988 else 'quote'
		return random.choice(temp)

	#=====================================================================================

	def RemovePlatformFromUserIDAndResolveNegatives(record):
		global nextUserID
		nameAndID = record['nodeUserID'].split('_',1)
		if nameAndID[0] in platformToEventTypeToEvent.keys():
			return nameAndID[1]
		elif record['nodeUserID'] == '-1.0':
			nextUserID += 1
			return str(nextUserID)
		else:
			return record['nodeUserID']

	def RemovePlatformFromColumn(record, columnName):
		nameAndID = record[columnName].split('_',1)
		if nameAndID[0] in platformToEventTypeToEvent.keys():
			return nameAndID[1]
		else:
			return record[columnName]

	print('\tmapping platform...')
	originalData['platform'] = originalData.apply(lambda x: x['nodeUserID'].split('_',1)[0],axis = 1)

	print('\tresolving userid, nodeid, parentid, and rootid...')
	print("\t\tCurrent NextUserID value: " + str(nextUserID))
	print("\tresolving userid values...")
	originalData['nodeUserID'] = originalData.apply(lambda x: RemovePlatformFromUserIDAndResolveNegatives(x), axis = 1)	
	print("\t\tCurrent NextUserID value: " + str(nextUserID))
	print("\t\tCurrent NextNodeID value: " + str(nextNodeID))
	print('\tresolving nodeid values...')
	originalData['nodeID'] = originalData.apply(lambda x: RemovePlatformFromColumn(x,'nodeID'), axis = 1)
	print("\t\tCurrent NextNodeID value: " + str(nextNodeID))
	print('\tresolving parentid values...')
	originalData['parentID'] = originalData.apply(lambda x: RemovePlatformFromColumn(x,'parentID'), axis = 1)
	print("\t\tCurrent NextNodeID value: " + str(nextNodeID))
	print('\tresolving rootid values...')
	originalData['rootID'] = originalData.apply(lambda x: RemovePlatformFromColumn(x,'rootID'), axis = 1)
	print("\t\tCurrent NextNodeID value: " + str(nextNodeID))

	print('\t\t nodeID')
	temp = originalData.loc[originalData['nodeID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))


	#fix github nodeID

	#resolve github seperately
	nextNodeID += 1
	gitNegOneValue = nextNodeID
	nextNodeID += 1
	def ReplaceGitHubNegOneRoot(x):
		global nextNodeID
		if (x['platform'] == "github") and (x["rootID"] == "-1.0"):
			return gitNegOneValue
		return x['rootID']
	
	originalData["rootID"] = originalData.apply(lambda x: ReplaceGitHubNegOneRoot(x),axis=1)
	originalData["parentID"] = originalData.apply(lambda x: x["rootID"] if x["platform"]=="github" else x["parentID"],axis =1 )
	originalData["nodeID"] = originalData.apply(lambda x: x["rootID"] if x["platform"]=="github" else x["nodeID"],axis =1 )

	print('\t\t nodeID')
	temp = originalData.loc[originalData['nodeID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))

	print('\tmappinging actionType...')
	originalData['actionType'] = originalData.apply(lambda x: MapEventUniformlyRandom(x),axis=1)

	print('\tcreating node to platform map...')
	node_to_plat = originalData.set_index('rootID').to_dict()['platform']
	node_to_plat.update(originalData.set_index('parentID').to_dict()['platform'])
	node_to_plat.update(originalData.set_index('nodeID').to_dict()['platform'])

	# -- Process INFORMATION ID --
	print('\treading infoid list file...')
	print('\t\tFile Name: ' + in_Params.nodelist_file)
	nodeList = list( pd.read_csv(in_Params.nodelist_file).iloc[:,0] )
	print('\tapplying random infoids to empty infoid values...')
	def PickRandomInfoIDIfEmpty(record):
		setOfinfoIDs = list(set(eval(record['informationID'])))
		if len(setOfinfoIDs) == 1 and setOfinfoIDs[0] == '-1.0':
			return [ nodeList[random.randint(0,len(nodeList)-1)] ]
		if record['informationID'] == '[]' or record['informationID'] == '-1.0' or record['informationID'] == '':
			return [ nodeList[random.randint(0,len(nodeList)-1)] ]
		return record['informationID']
	print(originalData)
	originalData['informationID'] = originalData.apply(lambda x: PickRandomInfoIDIfEmpty(x), axis = 1)
	
	print('\titerating the dataframe to multiplicate the multiple-infoid-valued rows...')
	print('infoids ' + str(originalData['informationID'].astype(str).unique().shape[0]))
	NewRows = []
	setOfReqNodes = set(nodeList)
	for index,row in originalData.iterrows():
		temp = str(row['informationID'])
		if temp[0] == '[' and temp != '[]':
			# check if the list is single token
			for iid in list(set(eval(temp))):
				if iid in setOfReqNodes:
					NewRows.append([ row['nodeTime'], row['nodeUserID'], row['actionType'], row['nodeID'], row['parentID'], row['rootID'], iid, row['platform']])
				else:
					print('{} is not in the set of required nodes'.format(iid))
		else:
			print("Bad InfoID")
			print(temp)
	
	originalData = pd.DataFrame(NewRows, columns=['nodeTime','nodeUserID','actionType','nodeID','parentID','rootID','informationID','platform'])
	print('infoids ' + str(originalData['informationID'].unique().shape[0]))
	print(originalData)
		
	# Verification
	print("\tVERIFICATION...")

	def ValidateTimeString(record):
		matchObj = re.fullmatch(r'^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}',str(record['nodeTime']))
		if matchObj == None:
			return 1
		else:
			return 0
	
	print('\t\t nodeTime')
	temp = originalData.apply(lambda x: ValidateTimeString(x), axis = 1).sum()
	print('\t\t\t Mismatching String Count : \t' + str(temp))
	thislog['nodeTime_msc'] = temp
	
	print('\t\t nodeUserID')
	temp = originalData.loc[originalData['nodeUserID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['nodeUserID_n1c'] = temp

	print('\t\t actionType')
	temp = originalData.loc[originalData['actionType'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['actionType_n1c'] = temp
	temp = originalData.loc[originalData['actionType'] == 'UNKNOWN-EVENT'].shape[0]
	print('\t\t\t UNKNOWN-EVENT Count : ' + str(temp))
	thislog['actionType_uec'] = temp

	print('\t\t nodeID')
	temp = originalData.loc[originalData['nodeID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['nodeID_n1c'] = temp

	print('\t\t parentID')
	temp = originalData.loc[originalData['parentID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['parentID_n1c'] = temp

	print('\t\t rootID')
	temp = originalData.loc[originalData['rootID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['rootID_n1c'] = temp

	print('\t\t informationID')
	set_of_nodeids = list(set(nodeList))
	def VerifyInfoID(record):
		if type(record['informationID']) == str:
			if len(record['informationID']) > 0 and record['informationID'][0] == '[' :
				return 1
			else:
				return 0
		return 1
	
	unknownInfoIds = {}
	def ValidateInfoID(record):
		if type(record['informationID']) == str:
			if not (record['informationID'] in set_of_nodeids):
				unknownInfoIds[record['informationID']] = 1
				return 1
		return 0
	
	temp = originalData['informationID'].unique().shape[0]
	print('\t\t\t Unique InfoID count : \t' + str(temp))
	thislog['informationID_uidc'] = temp
	temp = originalData.loc[originalData['informationID'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['informationID_n1c'] = temp
	temp = originalData.apply(lambda x: VerifyInfoID(x), axis = 1).sum()
	print('\t\t\t Verification Failed InfoID Count : \t' + str(temp))
	thislog['informationID_vmic'] = temp
	temp = originalData.apply(lambda x: ValidateInfoID(x), axis = 1).sum()
	print('\t\t\t Validation Failed InfoID Count : \t' + str(temp))
	print('\t\t\t Unknown InfoIDs : ' + str(len(unknownInfoIds.keys())))
	thislog['informationID_mic'] = temp
	thislog['informationID_uiidc'] = len(unknownInfoIds.keys())
	thislog['informationID_uiids'] = list(unknownInfoIds.keys())

	print('\t\t platform')
	temp = originalData.loc[originalData['platform'] == '-1.0'].shape[0]
	print('\t\t\t -1.0 Count : \t' + str(temp))
	thislog['platform_n1c'] = temp
	temp = originalData.loc[ (originalData['platform'] != 'github') & (originalData['platform'] != 'reddit') & (originalData['platform'] != 'twitter') & (originalData['platform'] != 'telegram') & (originalData['platform'] != 'youtube')].shape[0]
	print('\t\t\t Mismatch platform Count : \t' + str(temp) )
	thislog['platform_mpc'] = temp

	# print('\t\t has_URL')
	# temp = originalData.loc[(originalData['has_URL'] != 0) & (originalData['has_URL'] != 1)].shape[0]
	# print('\t\t\t not 1 or 0 Count : \t' + str(temp))
	# thislog['has_URL'] = temp

	# print('\t\t links_to_external')
	# temp = originalData.loc[(originalData['links_to_external'] != 0) & (originalData['links_to_external'] != 1)].shape[0]
	# print('\t\t\t not 1 or 0 Count : \t' + str(temp))
	# thislog['links_to_external'] = temp

	# print('\t\t domain_linked')
	# temp = originalData.loc[originalData['domain_linked'] == '-1.0'].shape[0]
	# print('\t\t\t -1.0 Count : \t' + str(temp))
	# thislog['domain_linked'] = temp

	print('\twriting output files...')
	# Write Weird 1st line !!!
	TheFirstLine = '{"identifier": "' + in_Params.IdentifierStr + '", "team": "ucf-garibay", "scenario": "'+ str(in_Params.ScenarioNo) + '" }\n'
	with open(file_eventLog,"w") as f:
		f.write(TheFirstLine)

	# Write The Events
	with open(file_eventLog,"a") as f: 
		originalData.nodeTime = originalData.nodeTime.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ'))
		for rec in originalData.to_dict(orient='records'):
			json.dump(rec,f)
			f.write('\n')

	#-----------------------
	x = { \
		'time_executed' : thisDT.strftime("%m/%d/%Y %H:%M:%S:%f"), \
		'variables' : [ \
			{\
				'name':'Scenario', \
				'value': str(in_Params.ScenarioNo) \
			},
			{\
				'name':'commit.sha', \
				'value': str(in_Params.CommitSha) \
			},
			{\
				'name':'start.time', \
				'value': str(in_Params.Sim_StartTime) \
			},
			{\
				'name':'end.time', \
				'value': str(in_Params.Sim_EndTime) \
			},
			{\
				'name':'group.description', \
				'value': str(in_Params.GroupDesc) \
			},
			{\
				'name':'run.group', \
				'value': str(in_Params.RunGroup) \
			},
			{\
				'name':'run.by', \
				'value': str(in_Params.RunBy) \
			},
			{\
				'name':'run.number', \
				'value': str(in_Params.RunNumber) \
			},
			{\
				'name':'run.type', \
				'value': 'single' # Options: 'single', 'bspace', 'bsearch', 'other' \
			},
			{\
				'name':'platform', \
				'value': 'python' \
			},
			{\
				'name':'model_name', \
				'value': in_Params.IdentifierStr \
			},
			{\
				'name':'original_output_file', \
				'value': str(in_Params.file_csv_eventlog) \
			},
			{\
				'name':'original_output_file_S3Loc', \
				'value': str(in_Params.file_csv_S3Location) \
			},
			{\
				'name':'model_MemoryDepth', \
				'value': str(in_Params.Model_MemoryDepth) \
			},
			{\
				'name':'model_OverloadFactor', \
				'value': str(in_Params.Model_OverloadFactor) \
			},
		], \
		'model_name' : in_Params.IdentifierStr \
		}

	with open(file_metadata,'w') as f:
		json.dump(x,f,indent=4)

	print("\nOutput Files:")
	print("\t" + file_eventLog)
	print("\t" + file_metadata)

	thislog['eventlogjson'] = file_eventLog
	thislog['mynote'] = in_Params.mynote
	if in_Params.mpLogQueue != None:
		in_Params.mpLogQueue.put(thislog)

def MainMethod():
	if len(sys.argv) < 2:
		print("Please provide parameter file path as a command line argument.")
		sys.exit()
	params = Parameters(sys.argv[1])
	print(params)
	MultiRun(params)

if __name__ == "__main__":
	MainMethod()
	#cProfile.run('MainMethod()')

