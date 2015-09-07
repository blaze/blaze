#!/usr/bin/env python

from __future__ import absolute_import

import os
import sys
import argparse

import yaml

from odo import resource
from odo.utils import ignoring

from .server import Server, DEFAULT_PORT

try:
    import __builtin__ as builtins
except ImportError:
    import builtins


__all__ = 'spider', 'from_yaml'


def _spider(resource_path, ignore, followlinks, hidden):
    resources = {}
    for filename in (os.path.join(resource_path, x)
                     for x in os.listdir(resource_path)):
        basename = os.path.basename(filename)
        if (basename.startswith(os.curdir) and not hidden or
                os.path.islink(filename) and not followlinks):
            continue
        if os.path.isdir(filename):
            new_resources = _spider(filename, ignore=ignore,
                                    followlinks=followlinks, hidden=hidden)
            if new_resources:
                resources[basename] = new_resources
        else:
            with ignoring(*ignore):
                resources[basename] = resource(filename)
    return resources


def spider(path, ignore=(ValueError, NotImplementedError), followlinks=True,
           hidden=False):
    """Traverse a directory and call ``odo.resource`` on its contentso

    Parameters
    ----------
    path : str
        Path to a directory of resources to load
    ignore : tuple of Exception, optional
        Ignore these exceptions when calling resource
    followlinks : bool, optional
        Follow symbolic links
    hidden : bool, optional
        Load hidden files

    Returns
    -------
    dict
        Possibly nested dictionary of containing basenames mapping to resources
    """
    return {
        os.path.basename(path): _spider(path, ignore=ignore,
                                        followlinks=followlinks,
                                        hidden=hidden)
    }


def from_yaml(path, ignore=(ValueError, NotImplementedError), followlinks=True,
              hidden=False):
    """Construct a dictionary of resources from a YAML specification.

    Parameters
    ----------
    path : str
        Path to a YAML specification of resources to load
    ignore : tuple of Exception, optional
        Ignore these exceptions when calling resource
    followlinks : bool, optional
        Follow symbolic links
    hidden : bool, optional
        Load hidden files

    Returns
    -------
    dict
        A dictionary mapping top level keys in a YAML file to resources.

    See Also
    --------
    spider : Traverse a directory tree for resources
    """
    resources = {}
    for name, info in yaml.load(path.read()).items():
        if 'source' not in info:
            raise ValueError('source key not found for data source named %r' %
                             name)
        source = info['source']
        if os.path.isdir(source):
            resources[name] = spider(os.path.expanduser(source),
                                     ignore=ignore,
                                     followlinks=followlinks,
                                     hidden=hidden)
        else:
            resources[name] = resource(source, dshape=info.get('dshape'))
    return resources


def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('path', type=argparse.FileType('r'), nargs='?',
                   default=sys.stdin,
                   help='A YAML file specifying the resources to load')
    p.add_argument('-p', '--port', type=int, default=DEFAULT_PORT,
                   help='Port number')
    p.add_argument('-H', '--host', type=str, default='127.0.0.1',
                   help='Host name. Use 0.0.0.0 to listen on all public IPs')
    p.add_argument('-l', '--follow-links', action='store_true',
                   help='Follow links when listing files')
    p.add_argument('-e', '--ignored-exception', nargs='+',
                   default=['Exception'],
                   help='Exceptions to ignore when calling resource on a file')
    p.add_argument('-d', '--hidden', action='store_true',
                   help='Call resource on hidden files')
    p.add_argument('-D', '--debug', action='store_true',
                   help='Start the Flask server in debug mode')
    return p.parse_args()


def _main():
    args = _parse_args()
    ignore = tuple(getattr(builtins, e) for e in args.ignored_exception)
    resources = from_yaml(args.path,
                          ignore=ignore,
                          followlinks=args.follow_links,
                          hidden=args.hidden)
    Server(resources).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    _main()
