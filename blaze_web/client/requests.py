# -*- coding: utf-8 -*-

import urllib, urllib2, json

def get_remote_datashape(url):
    """Gets the datashape of a remote array URL."""
    response = urllib2.urlopen(url + '?r=datashape')
    return response.read()

def get_remote_json(url):
    """Gets the JSON data of a remote array URL."""
    response = urllib2.urlopen(url + '?r=data.json')
    return response.read()

def create_remote_session(base_url):
    """Creates a compute session rooted on the remote array URL."""
    params = [('r', 'create_session')]
    response = urllib2.urlopen(base_url, urllib.urlencode(params))
    return json.loads(response.read())

def close_remote_session(session_url):
    """Closes the remote compute session."""
    params = [('r', 'close_session')]
    response = urllib2.urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())

def add_computed_fields(session_url, url, fields, rm_fields, fnname):
    """Creates a new remote array with the added computed fields."""
    reqdata = {
            "input": str(url),
            "fields": [[str(name), str(dt), str(expr)]
                    for name, dt, expr in fields]
        }
    if len(rm_fields) > 0:
        reqdata['rm_fields'] = [str(name) for name in rm_fields]
    if fnname is not None:
        reqdata['fnname'] = str(fnname)
    params = [('r', 'add_computed_fields'),
              ('json', json.dumps(reqdata))]
    response = urllib2.urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())

