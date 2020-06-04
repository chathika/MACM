from collections import OrderedDict

event_types=OrderedDict({
    "creation": ["CreateEvent","tweet","post","Post","video"],
    "contribution": ['IssueCommentEvent', 'PullRequestEvent',
    'GollumEvent', 'PullRequestReviewCommentEvent', 'PushEvent', 
    'IssuesEvent', 'CommitCommentEvent',"DeleteEvent","reply","quote","message","comment","Comment"],
    "sharing": ["ForkEvent","WatchEvent", 'ReleaseEvent', 'MemberEvent', 'PublicEvent',"retweet"]
})

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