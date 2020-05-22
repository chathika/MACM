from collections import OrderedDict

event_types=OrderedDict({
    "creation": ["CreateEvent","tweet","post","Post","video"],
    "contribution": ['IssueCommentEvent', 'PullRequestEvent',
    'GollumEvent', 'PullRequestReviewCommentEvent', 'PushEvent', 
    'IssuesEvent', 'CommitCommentEvent',"DeleteEvent","reply","quote","message","comment","Comment"],
    "sharing": ["ForkEvent","WatchEvent", 'ReleaseEvent', 'MemberEvent', 'PublicEvent',"retweet"]
})


def getEventDictionary():
    return event_types


def getEventTypeIdx(action):    
    for idx,name in enumerate(getEventDictionary().keys()):
        if action in getEventDictionary()[name]:
            return idx    

def getEventTypes():
    return list(getEventDictionary().keys())



et=len(getEventTypes())
creation_idx = list(getEventDictionary().keys()).index("creation")