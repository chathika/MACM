import pandas as pd
import random
import os
import numpy as np
import datetime
import json
import re

# Common Options
_multi_run = False

_ScenarioNo = 1
_Sim_StartTime = '2017.04.01'
_Sim_EndTime = '2018.03.31'
_CurrentSprint = 14
_CommitSha = '05be2aaf9e8d61ecc2b48cf0735028db2a3f0bce'
_output_directory_path = "/home/social-sim/MACMWorking/MACM/Output_manual"

_nodelist_file = '/home/social-sim/MACMWorking/MACM/DryRun/Inputs/cp3_dry_run_s1_nodelist_updated.txt'

_GroupDesc = 'MACMGPUv2.10'
_RunBy = 'Chathika'
_RunNumber = 0

# Multiple Files Optioins
_mr_csv_folder = '/home/social-sim/MACMWorking/MACM/FactorialRun/Original'

# Single File Options
_file_csv_eventlog = '/home/social-sim/MACMWorking/MACM/Output_manual/MACM_MMD10_Alpha0.8_2020-01-14 20_12_49.260452.csv'
_file_csv_S3Location = 'null'
_Model_MemoryDepth = 10
_Model_OverloadFactor = 0.8
_IdentifierStr = 'MACMGPU-sp' + str(_CurrentSprint)
_RunGroup = 'cp3_sp' + str(_CurrentSprint) + '_MACMGPU26'

def MultiRun():
	print("--Begining of Script--")
	if not _multi_run:
		Run(_file_csv_eventlog, _nodelist_file, _file_csv_S3Location, _ScenarioNo, _Sim_StartTime, _Sim_EndTime, _Model_MemoryDepth, _Model_OverloadFactor, _IdentifierStr, _CurrentSprint, _CommitSha, _RunGroup, _GroupDesc, _RunBy, _RunNumber, _output_directory_path)
	else:
		for f in os.listdir(_mr_csv_folder):
			print('-- Processing File :' + f + ' --')
			file_csv_eventlog = os.path.join(_mr_csv_folder,f)
			temp = re.search('MMD[0-9]*',f)
			Model_MemoryDepth = f[temp.span()[0] + 3 : temp.span()[1]]
			temp = re.search('Alpha[0-9.]*',f)
			Model_OverloadFactor = float(f[temp.span()[0] + 5 : temp.span()[1]])
			IdentifierStr = 'MACMGPUx' + str(Model_MemoryDepth) + '_' + ( "%.2f" % Model_OverloadFactor ).replace('.','_')
			RunGroup = 'cp3_sp' + str(_CurrentSprint) + '_' + IdentifierStr
			Run(file_csv_eventlog, _nodelist_file, _file_csv_S3Location, _ScenarioNo, _Sim_StartTime, _Sim_EndTime, Model_MemoryDepth, Model_OverloadFactor, IdentifierStr, _CurrentSprint, _CommitSha, RunGroup, _GroupDesc, _RunBy, _RunNumber, _output_directory_path)
	print("--End of Script--")

def Run(file_csv_eventlog, _nodelist_file, file_csv_S3Location, ScenarioNo, Sim_StartTime, Sim_EndTime, Model_MemoryDepth, Model_OverloadFactor, IdentifierStr, CurrentSprint, CommitSha, RunGroup, GroupDesc, RunBy, RunNumber, output_directory_path):
	if not os.path.exists(output_directory_path):
		print ("The Output directory path does not exist!")
		exit()

	thisDT = datetime.datetime.now()
	file_name_prefix = thisDT.strftime("%m-%d-%Y-%H-%M-%S-%f") + "-" + str(RunGroup) + "-" + str(RunNumber)
	file_eventLog = output_directory_path + "/" + file_name_prefix + "-MP-Sc" + str(ScenarioNo) + ".json"
	file_metadata = output_directory_path + "/" + file_name_prefix + "-xMetadata.json"

	print('\treading event log file...')
	originalData = pd.read_csv(file_csv_eventlog,dtype={'userID':str,'conversationID':str,'parentID':str,'nodeID':str},na_values=str)
	print('\trenaming columns...')
	originalData.rename(columns={'userID':'nodeUserID','action':'actionType','time':'nodeTime','conversationID':'rootID'},inplace=True)

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

	def RemovePlatformFromID(IDString):
		nameAndID = IDString.split('_',1)
		if nameAndID[0] == 'github' or \
		nameAndID[0] == 'reddit' or \
		nameAndID[0] == 'twitter' or \
		nameAndID[0] == 'telegram':
			return nameAndID[1]
		else:
			return IDString

	print('\tmapping platform...')
	originalData['platform'] = originalData.apply(lambda x: x['nodeUserID'].split('_')[0],axis = 1)
	print('\tmappinging actionType...')
	originalData['actionType'] = originalData.apply(lambda x: MapEventProbablyRandom(x),axis=1)
	#originalData['informationID'] = ""
	print('\tremoving platform ids from userid, nodeid, parentid, and rootid...')
	originalData['nodeUserID'] = originalData.nodeUserID.apply(lambda x: RemovePlatformFromID(x))
	originalData['nodeID'] = originalData.nodeID.apply(lambda x: RemovePlatformFromID(x))
	originalData['parentID'] = originalData.parentID.apply(lambda x: RemovePlatformFromID(x))
	originalData['rootID'] = originalData.rootID.apply(lambda x: RemovePlatformFromID(x))
	print('\tsetting url parameters...')
	originalData['urlDomain'] = np.empty((len(originalData),0)).tolist()
	originalData['has_URL'] = 0
	originalData['links_to_external'] = 0
	originalData['domain_linked'] = np.empty((len(originalData),0)).tolist()

	# -- Process INFORMATION ID --
	print('\treading infoid list file...')
	nodeList = pd.read_csv(_nodelist_file)
	print('\tsetting random info ids...')
	originalData['informationID'] = np.random.choice(nodeList[nodeList.columns[0]],originalData.shape[0],replace=True)
	#originalData['informationID'] = originalData['informationID'].apply(lambda x: nodeList.sample(1).iloc[0].tolist()[0])
	print('\tcalculating node to infoid and node to parent hash maps...')
	node_to_info = originalData.set_index('nodeID').to_dict()['informationID']
	node_to_parent = originalData.set_index('nodeID').to_dict()['parentID']
	def PickRootInfoID(record):
		if record['rootID'] in node_to_info.keys():
			return node_to_info[ record['rootID'] ]
		elif record['parentID'] in node_to_info.keys():
			currentParent = record['parentID']
			#print('\t\tsolving parent: ' + str(currentParent))
			while node_to_parent[currentParent] in node_to_info.keys():
				currentParent = node_to_parent[currentParent]
				#print('\t\t\tlooking for parent of : '+str(currentParent))
				if currentParent == node_to_parent[currentParent]:
					break
			#print('\t\t\tresolved to : ' + str(currentParent))
			return node_to_info[currentParent]
		else:
			return record['informationID']
	
	print('\tmapping infoids according to cascades...')
	#return originalData.loc[ originalData['nodeID'] == record['rootID'] , 'informationID' ].append(pd.Series([ record['informationID'] ])).iloc[0]
	originalData['informationID'] = originalData.apply(lambda x: PickRootInfoID(x), axis = 1)

	print('\twriting output files...')
	# Write Weird 1st line !!!
	TheFirstLine = '{"identifier": "' + IdentifierStr + '", "team": "ucf-garibay", "scenario": "'+ str(ScenarioNo) + '" }\n'
	with open(file_eventLog,"w") as f:
		f.write(TheFirstLine)

	# Write The Events
	with open(file_eventLog,"a") as f: 
		originalData.to_json(f,orient='records',lines=True)

	#-----------------------
	x = { \
		'time_executed' : thisDT.strftime("%m/%d/%Y %H:%M:%S:%f"), \
		'variables' : [ \
			{\
				'name':'Scenario', \
				'value': str(ScenarioNo) \
			},
			{\
				'name':'commit.sha', \
				'value': str(CommitSha) \
			},
			{\
				'name':'start.time', \
				'value': str(Sim_StartTime) \
			},
			{\
				'name':'end.time', \
				'value': str(Sim_EndTime) \
			},
			{\
				'name':'group.description', \
				'value': str(GroupDesc) \
			},
			{\
				'name':'run.group', \
				'value': str(RunGroup) \
			},
			{\
				'name':'run.by', \
				'value': str(RunBy) \
			},
			{\
				'name':'run.number', \
				'value': str(RunNumber) \
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
				'value': IdentifierStr \
			},
			{\
				'name':'original_output_file', \
				'value': str(file_csv_eventlog) \
			},
			{\
				'name':'original_output_file_S3Loc', \
				'value': str(file_csv_S3Location) \
			},
			{\
				'name':'model_MemoryDepth', \
				'value': str(Model_MemoryDepth) \
			},
			{\
				'name':'model_OverloadFactor', \
				'value': str(Model_OverloadFactor) \
			},
		], \
		'model_name' : IdentifierStr \
		}

	with open(file_metadata,'w') as f:
		json.dump(x,f,indent=4)

	print("\nOutput Files:")
	print("\t" + file_eventLog)
	print("\t" + file_metadata)

MultiRun()
