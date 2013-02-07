import os, glob
from os import path
import tempfile

from dynd import nd, ndt

def load_json_file_array(root, array_name):
    # Load the datashape
    dsfile = root + '.datashape'
    if not path.isfile(dsfile):
        dsfile = path.dirname(root) + '.datashape'
        if not path.isfile(dsfile):
            raise Exception('No datashape file found for array %s' % array_name)
    with open(dsfile) as f:
        dt = nd.dtype(f.read())

    # Load the JSON
    with open(root + '.json') as f:
        # TODO: Add stream support to parse_json for compressed JSON, etc.
        arr = nd.parse_json(dt, f.read())
    return arr

def load_json_directory_array(root, array_name):
    # Load the datashape
    dsfile = root + '.datashape'
    if not path.isfile(dsfile):
        raise Exception('No datashape file found for array %s' % array_name)
    with open(dsfile) as f:
        dt = nd.dtype(f.read())

    # Scan for JSON files, assuming they're just #.json
    # Sort them numerically
    files = sorted([(int(path.splitext(path.basename(x))[0]), x)
                    for x in glob.glob(path.join(root, '*.json'))])
    files = [x[1] for x in files]
    # Make an array with an extra fixed dimension, then
    # read a JSON file into each element of that array
    dt = ndt.make_fixedarray_dtype(dt, len(files))
    arr = nd.empty(dt)
    for i, fname in enumerate(files):
        with open(fname) as f:
            nd.parse_json(arr[i], f.read())
    arr.flag_as_immutable()
    return arr

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
        if not path.isfile(root + '.json') and not path.isdir(root):
            return None

        # If we've already read this array into cache, just return it
        if self.array_cache.has_key(root):
            print 'Returning cached array %s' % array_name
            return self.array_cache[root]

        if path.isfile(root + '.json'):
            print 'Loading array %s from file %s' % (array_name, root + '.json')
            arr = load_json_file_array(root, array_name)
        else:
            print 'Loading array %s from directory %s' % (array_name, root)
            arr = load_json_directory_array(root, array_name)
            
        self.array_cache[root] = arr
        return arr

    def create_session_dir(self):
        d = tempfile.mkdtemp(prefix='.session_', dir=self.root_dir)
        return os.path.basename(d), d
