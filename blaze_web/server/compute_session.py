import json
from blaze_web.common.blaze_url import split_array_base
from dynd import nd, ndt

class compute_session:
    def __init__(self, array_provider, base_url, array_name):
        self.array_provider = array_provider
        session_name, root_dir = array_provider.create_session_dir()
        self.session_name = array_name + '/' + session_name
        self.root_dir = root_dir
        self.array_name = array_name
        self.base_url = base_url

    def get_session_array(self):
        array_name, indexers = split_array_base(self.array_name)
        arr = self.array_provider(array_name)
        if arr is None:
            raise Exception('No Blaze Array named ' + array_name)

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
                'access' : 'no permission model yet'
            })
        return (content_type, body)

    def create_table_view(self, json_cmd):
        cmd = json.reads(json_cmd)
        ds = cmd['datashape']
        dt = nd.dtype(ds)
        ex = cmd['expressions']
        
        arr = self.get_session_array()
        print arr.dshape

        result = nd.empty_like(arr, dt)
        for fname in dt.field_names:
            print fname
        