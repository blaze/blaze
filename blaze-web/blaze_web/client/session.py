# -*- coding: utf-8 -*-

from requests import create_remote_session, close_remote_session, \
        add_computed_fields, sort, groupby
from rarray import rarray

class session:
    def __init__(self, root_url):
        """
        Creates a remote Blaze compute session with the
        requested Blaze remote array as the root.
        """
        self.root_url = root_url
        j = create_remote_session(root_url)
        self.session_url = j['session']
        self.server_version = j['version']

        print('Remote Blaze session created at %s' % root_url)
        print('Remote DyND-Python version: %s' % j['dynd_python_version'])
        print('Remote DyND version: %s' % j['dynd_version'])

    def __repr__(self):
        return 'Blaze Remote Compute Session\n' + \
                        ' root url: ' + self.root_url + '\n' \
                        ' session url: ' + self.session_url + '\n' + \
                        ' server version: ' + self.server_version + '\n'

    def add_computed_fields(self, arr, fields, rm_fields=[], fnname=None):
        """
        Adds one or more new fields to a struct array.
        
        Each field_expr in 'fields' is a string/ast fragment
        which is called using eval, with the input fields
        in the locals and numpy/scipy in the globals.
    
        arr : rarray
            A remote array on the server.
        fields : list of (field_name, field_type, field_expr)
            These are the fields which are added to 'n'.
        rm_fields : list of string, optional
            For fields that are in the input, but have no expression,
            this removes them from the output struct instead of
            keeping the value.
        fnname : string, optional
            The function name, which affects how the resulting
            deferred expression dtype is printed.
        """
        j = add_computed_fields(self.session_url,
                                   arr.url, fields,
                                   rm_fields, fnname)
        return rarray(j['output'], j['dshape'])
    
    def sort(self, arr, field):
        j = sort(self.session_url, arr.url, field)
        return rarray(j['output'], j['dshape'])

    def groupby(self, arr, fields):
        """
        Applies a groupby to a struct array based on selected fields.
        
        arr : rarray
            A remote array on the server.
        fields : list of field names
            These are the fields which are used for grouping.
        
        Returns a tuple of the groupby result and the groups.
        """
        j = groupby(self.session_url, arr.url, fields)
        return (
            rarray(j['output_gb'], j['dshape_gb']),
            rarray(j['output_groups'], j['dshape_groups']))

    def close(self):
        close_remote_session(self.session_url)
        self.session_url = None
