import six
from six import StringIO
from six.moves import urllib
from contextlib import contextmanager
if six.PY2:
    import snakebite
    from snakebite.client import Client

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 9000

def host_port_path(hdfs_filepath):
    """
    Get the hostname, path, and port number from 
    an HDFS URI. Supply defaults if not given
    hdfs_filepath : string
      Either a full URI or just a path on the filesystem. 
      If the latter, return the defaults
    """
    parse_result = urllib.parse.urlparse(hdfs_filepath)
    if parse_result.netloc:
        if ':' in parse_result.netloc:
            host, port = parse_result.netloc.split(':')
        else:
            host = parse_result.netloc
            port = DEFAULT_PORT
    else:
        host = DEFAULT_HOSTNAME
        port = DEFAULT_PORT

    return host, int(port), parse_result.path


def hdfs_isfile(path, host=None, port=None):
    """ 
    A drop-in replacement for os.path.isfile. Returns true
    if the file exists on the given HDFS system. False otherwise.
    path : string
        A path string for the file. Full URI or just local path
        Ex: /usr/joe/stuff.txt
    host : string
        hostname for the HDFS system
    port : int
        Port number for the HDFS system
    """
    ahost, aport, apath = host_port_path(path)
    if not host:
        host = ahost
    if not port:
        port = aport
    path = apath

    client = Client(host, port, use_trash=False)
    try:
        return client.test(path, exists=True)
    except:
        return False

@contextmanager
def hdfs_open(path, mode='r', host=None, port=None):
    """ 
    A context manager that provides functionality equivalent to the 
    standard open function.
    path : string
        A path string for the file. (not the full URI):
        Ex: /usr/joe/stuff.txt
    host : string
        hostname for the HDFS system
    port : int
        Port number for the HDFS system
    mode : string
        The mode for opening the file. It's likely that the 
        only good choice here is 'r'
    """
    ahost, aport, apath = host_port_path(path)
    if not host:
        host = ahost
    if not port:
        port = aport
    path = apath

    client = Client(host, port, use_trash=False)
    for files in client.cat([path]): 
        for text in files:
            sb = StringIO(text)

    yield sb
    sb.seek(0)
