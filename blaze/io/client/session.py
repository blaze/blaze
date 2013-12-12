# -*- coding: utf-8 -*-

from __future__ import absolute_import
from .requests import create_remote_session, close_remote_session, \
        add_computed_fields, make_computed_fields, sort, groupby
from .rarray import rarray

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

    def make_computed_fields(self, arr, replace_undim, fields, fnname=None):
        """
        Creates an array with the requested computed fields.
        If replace_undim is positive, that many uniform dimensions
        are provided into the field expressions, so the
        result has fewer dimensions.

        arr : rarray
            A remote array on the server.
        replace_undim : integer
            The number of uniform dimensions to leave in the
            input going to the fields. For example if the
            input has shape (3,4,2) and replace_undim is 1,
            the result will have shape (3,4), and each operand
            provided to the field expression will have shape (2).
        fields : list of (field_name, field_type, field_expr)
            These are the fields which are added to 'n'.
        fnname : string, optional
            The function name, which affects how the resulting
            deferred expression dtype is printed.
        """
        j = make_computed_fields(self.session_url,
                                   arr.url, replace_undim, fields,
                                   fnname)
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
