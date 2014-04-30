from __future__ import absolute_import, division, print_function

import urllib
import json

from ... import compatibility

if compatibility.PY2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

def get_remote_datashape(url):
    """Gets the datashape of a remote array URL."""
    response = urlopen(url + '?r=datashape')
    return response.read().decode('utf8')

def get_remote_json(url):
    """Gets the JSON data of a remote array URL."""
    response = urlopen(url + '?r=data.json')
    return response.read()

def create_remote_session(base_url):
    """Creates a compute session rooted on the remote array URL."""
    params = [('r', 'create_session')]
    response = urlopen(base_url, urllib.urlencode(params))
    return json.loads(response.read())

def close_remote_session(session_url):
    """Closes the remote compute session."""
    params = [('r', 'close_session')]
    response = urlopen(session_url, urllib.urlencode(params))
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
    response = urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())

def make_computed_fields(session_url, url, replace_undim, fields, fnname):
    """Creates a new remote array with the computed fields."""
    reqdata = {
            "input": str(url),
            "replace_undim": int(replace_undim),
            "fields": [[str(name), str(dt), str(expr)]
                    for name, dt, expr in fields]
        }
    if fnname is not None:
        reqdata['fnname'] = str(fnname)
    params = [('r', 'make_computed_fields'),
              ('json', json.dumps(reqdata))]
    response = urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())

def sort(session_url, url, field):
    """Creates a new remote array which is sorted by field."""
    reqdata = {
        "input": str(url),
        "field": field
        }
    params = [('r', 'sort'),
              ('json', json.dumps(reqdata))]
    response = urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())

def groupby(session_url, url, fields):
    reqdata = {
        "input": str(url),
        "fields": fields
        }
    params = [('r', 'groupby'),
              ('json', json.dumps(reqdata))]
    response = urlopen(session_url, urllib.urlencode(params))
    return json.loads(response.read())
