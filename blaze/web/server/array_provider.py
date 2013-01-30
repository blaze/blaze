import os
from os import path

from dynd import nd, ndt

class json_array_provider:
    def __init__(self, root_dir):
        if not path.isdir(root_dir):
            raise ValueError('%s is not a valid directory' % root_dir)
        self.root_dir = root_dir
        self.array_cache = {}

    def __call__(self, array_name):
        # First check that the .json file at the requested address exists
        if array_name[0] == '/':
            array_name = array_name[1:]
        root = path.join(self.root_dir, array_name)
        jfile = jfile + '.json'
        if not path.isfile(jfile):
            return None

        # If we've already read this array into cache, just return it
        if self.array_cache.has_key(jfile):
            return self.array_cache[jfile]

        # Search for the datashape file of this array
        dsfile = None
        if path.isfile(root + '.datashape'):
            dsfile = root + '.datashape'
        else:
            an_components = array_name.split('/')
            l = len(an_components)
            if l > 1:
                for i in range(1, l):
                    partial_root = path.join(self.root_dir, '/'.join(an_components[:(l-i)]))
                    if path.isfile(partial_root + '.datashape'):
                        dsfile = partial_root + '.datashape'
                        break
        if dsfile is None:
            raise Exception('No datashape file found for array %s' % array_name)
        with open(dsfile) as f:
            dt = nd.dtype(f.read())

        with open(jfile) as f:
            # TODO: Add stream support to parse_json for compressed JSON, etc.
            return nd.parse_json(dt, f.read())
