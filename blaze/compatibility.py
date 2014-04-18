
import sys
PY3 = sys.version_info[0] > 2

if PY3:
    from urllib.request import urlopen

else:
    from urllib2 import urlopen
