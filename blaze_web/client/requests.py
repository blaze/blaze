# -*- coding: utf-8 -*-

import urllib2

def get_remote_datashape(url):
    response = urllib2.urlopen(url + '?r=datashape')
    return response.read()

def get_remote_json(url):
    response = urllib2.urlopen(url + '?r=data.json')
    return response.read()
