# -*- coding: utf-8 -*-

import urllib2

def get_remote_datashape(url):
    response = urllib2.urlopen(url + '?r=datashape')
    print response.getcode()
    return response.read()