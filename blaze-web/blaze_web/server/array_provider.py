import os, glob, shutil
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
    dt = ndt.make_fixed_dim_dtype(len(files), dt)
    arr = nd.empty(dt)
    for i, fname in enumerate(files):
        with open(fname) as f:
            nd.parse_json(arr[i], f.read())
    arr.flag_as_immutable()
    return arr

def load_json_file_list_array(root, array_name):
    # Load the datashape
    dsfile = root + '.datashape'
    if not path.isfile(dsfile):
        raise Exception('No datashape file found for array %s' % array_name)
    with open(dsfile) as f:
        dt = nd.dtype(f.read())
    
    # Scan for JSON files -- no assumption on file suffix
    
    #open list of files and load into python list
    files = root + '.files'
    with open(files) as f:
        l_files = [fs.strip() for fs in f]

    # Make an array with an extra fixed dimension, then
    # read a JSON file into each element of that array
    dt = ndt.make_fixed_dim_dtype(len(l_files), dt)
    arr = nd.empty(dt)
    for i, fname in enumerate(l_files):
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
        self.session_dirs = {}

    def __call__(self, array_name):
        # First check that the .json file at the requested address exists
        root = path.join(self.root_dir, array_name[1:])
        if not path.isfile(root + '.json') and \
                        not path.isfile(root + '.deferred.json') and \
                        not path.isfile(root + '.files') and \
                        not path.isdir(root):
            return None

        # If we've already read this array into cache, just return it
        print('Cache has keys %s' % self.array_cache.keys())
        print('Checking cache for %s' % array_name)
        if self.array_cache.has_key(array_name):
            print 'Returning cached array %s' % array_name
            return self.array_cache[array_name]

        if path.isfile(root + '.json'):
            print('Loading array %s from file %s' %
                            (array_name, root + '.json'))
            arr = load_json_file_array(root, array_name)
        elif path.isfile(root + '.deferred.json'):
            print('Loading deferred array %s from file %s' %
                            (array_name, root + '.deferred.json'))
            with open(root + '.deferred.json') as f:
                print(f.read())
            raise RuntimeError('TODO: Deferred loading not implemented!')       
        elif path.isfile(root + '.files'):
            print ('Loading files from file list: %s' % (root + '.files'))
            arr = load_json_file_list_array(root, array_name)
        else:
            print 'Loading array %s from directory %s' % (array_name, root)
            arr = load_json_directory_array(root, array_name)
            
        self.array_cache[array_name] = arr
        return arr

    def create_session_dir(self):
        d = tempfile.mkdtemp(prefix='.session_', dir=self.root_dir)
        session_name = '/' + os.path.basename(d)
        if type(session_name) is unicode:
            session_name = session_name.encode('utf-8')
        self.session_dirs[session_name] = d
        return session_name, d

    def delete_session_dir(self, session_name):
        shutil.rmtree(self.session_dirs[session_name])
        del self.session_dirs[session_name]

    def create_deferred_array_filename(self, session_name,
                                       prefix, cache_array):
        d = tempfile.mkstemp(suffix='.deferred.json', prefix=prefix,
                             dir=self.session_dirs[session_name], text=True)
        array_name = os.path.basename(d[1])
        array_name = session_name + '/' + array_name[:array_name.find('.')]
        if type(array_name) is unicode:
            array_name = array_name.encode('utf-8')

        if cache_array is not None:
            self.array_cache[array_name] = cache_array
        
        return (os.fdopen(d[0], "w"), array_name, d[1])
