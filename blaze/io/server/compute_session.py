from __future__ import absolute_import, division, print_function

import json

from blaze.catalog.blaze_url import split_array_base
import dynd
from dynd import nd
from dynd.nd import as_numpy
from blaze import array


class compute_session:
    def __init__(self, base_url, array_name):
        session_name, root_dir = array_provider.create_session_dir()
        self.session_name = session_name
        self.root_dir = root_dir
        self.array_name = array_name
        self.base_url = base_url

    def get_session_array(self, array_name = None):
        if array_name is None:
            array_name = self.array_name
        array_root, indexers = split_array_base(array_name)
        arr = self.array_provider(array_root)
        if arr is None:
            raise Exception('No Blaze Array named ' + array_root)

        for i in indexers:
            if type(i) in [slice, int, tuple]:
                arr = arr[i]
            else:
                arr = getattr(arr, i)
        return arr

    def creation_response(self):
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session' : self.base_url + self.session_name,
                'version' : 'prototype',
                'dynd_python_version': dynd.__version__,
                'dynd_version' : dynd.__libdynd_version__,
                'access' : 'no permission model yet'
            })
        return (content_type, body)

    def close(self):
        print('Deleting files for session %s' % self.session_name)
        self.array_provider.delete_session_dir(self.session_name)
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session': self.base_url + self.session_name,
                'action': 'closed'
            })
        return (content_type, body)

    def sort(self, json_cmd):
        import numpy as np
        print ('sorting')
        cmd = json.loads(json_cmd)
        array_url = cmd.get('input', self.base_url + self.array_name)
        if not array_url.startswith(self.base_url):
            raise RuntimeError('Input array must start with the base url')
        array_name = array_url[len(self.base_url):]
        field = cmd['field']
        arr = self.get_session_array(array_name)
        nparr = as_numpy(arr)
        idxs = np.argsort(nparr[field])
        res = nd.ndobject(nparr[idxs])
        defarr = self.array_provider.create_deferred_array_filename(
                        self.session_name, 'sort_', res)
        dshape = nd.dshape_of(res)
        defarr[0].write(json.dumps({
                'dshape': dshape,
                'command': 'sort',
                'params': {
                    'field': field,
                }
            }))
        defarr[0].close()
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session': self.base_url + self.session_name,
                'output': self.base_url + defarr[1],
                'dshape': dshape
            })
        return (content_type, body)

    def groupby(self, json_cmd):
        print('GroupBy operation')
        cmd = json.loads(json_cmd)
        array_url = cmd.get('input', self.base_url + self.array_name)
        if not array_url.startswith(self.base_url):
            raise RuntimeError('Input array must start with the base url')
        array_name = array_url[len(self.base_url):]
        fields = cmd['fields']

        arr = self.get_session_array(array_name)[...].ddesc.dynd_arr()

        # Do the groupby, get its groups, then
        # evaluate it because deferred operations
        # through the groupby won't work well yet.
        res = nd.groupby(arr, nd.fields(arr, *fields))
        groups = res.groups
        res = res.eval()

        # Write out the groupby result
        defarr_gb = self.array_provider.create_deferred_array_filename(
                        self.session_name, 'groupby_', array(res))
        dshape_gb = nd.dshape_of(res)
        defarr_gb[0].write(json.dumps({
                'dshape': dshape_gb,
                'command': 'groupby',
                'params': {
                    'fields': fields
                }
            }))
        defarr_gb[0].close()

        # Write out the groups
        defarr_groups = self.array_provider.create_deferred_array_filename(
                        self.session_name, 'groups_', groups)
        dshape_groups = nd.dshape_of(groups)
        defarr_groups[0].write(json.dumps({
                'dshape': dshape_groups,
                'command': 'groupby.groups',
                'params': {
                    'fields': fields
                }
            }))
        defarr_groups[0].close()

        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session': self.base_url + self.session_name,
                'output_gb': self.base_url + defarr_gb[1],
                'dshape_gb': dshape_gb,
                'output_groups': self.base_url + defarr_groups[1],
                'dshape_groups': dshape_groups
            })
        return (content_type, body)

    def add_computed_fields(self, json_cmd):
        print('Adding computed fields')
        cmd = json.loads(json_cmd)
        array_url = cmd.get('input', self.base_url + self.array_name)
        if not array_url.startswith(self.base_url):
            raise RuntimeError('Input array must start with the base url')
        array_name = array_url[len(self.base_url):]
        fields = cmd['fields']
        rm_fields = cmd.get('rm_fields', [])
        fnname = cmd.get('fnname', None)

        arr = self.get_session_array(array_name).ddesc.dynd_arr()

        res = nd.add_computed_fields(arr, fields, rm_fields, fnname)
        defarr = self.array_provider.create_deferred_array_filename(
                        self.session_name, 'computed_fields_', array(res))
        dshape = nd.dshape_of(res)
        defarr[0].write(json.dumps({
                'dshape': dshape,
                'command': 'add_computed_fields',
                'params': {
                    'fields': fields,
                    'rm_fields': rm_fields,
                    'fnname': fnname
                }
            }))
        defarr[0].close()
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session': self.base_url + self.session_name,
                'output': self.base_url + defarr[1],
                'dshape': dshape
            })
        return (content_type, body)

    def make_computed_fields(self, json_cmd):
        print('Adding computed fields')
        cmd = json.loads(json_cmd)
        array_url = cmd.get('input', self.base_url + self.array_name)
        if not array_url.startswith(self.base_url):
            raise RuntimeError('Input array must start with the base url')
        array_name = array_url[len(self.base_url):]
        fields = cmd['fields']
        replace_undim = cmd.get('replace_undim', 0)
        fnname = cmd.get('fnname', None)

        arr = self.get_session_array(array_name).ddesc.dynd_arr()

        res = nd.make_computed_fields(arr, replace_undim, fields, fnname)
        defarr = self.array_provider.create_deferred_array_filename(
                        self.session_name, 'computed_fields_', array(res))
        dshape = nd.dshape_of(res)
        defarr[0].write(json.dumps({
                'dshape': dshape,
                'command': 'make_computed_fields',
                'params': {
                    'fields': fields,
                    'replace_undim': replace_undim,
                    'fnname': fnname
                }
            }))
        defarr[0].close()
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session': self.base_url + self.session_name,
                'output': self.base_url + defarr[1],
                'dshape': dshape
            })
        return (content_type, body)
