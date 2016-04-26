#!/usr/bin/env python

from __future__ import absolute_import

import os
import sys
import argparse
import importlib

from contextlib import contextmanager

import yaml

from blaze.interactive import data as bz_data
from odo.utils import ignoring

from .server import Server, DEFAULT_PORT

try:
    import __builtin__ as builtins
except ImportError:
    import builtins


__all__ = 'data_spider', 'from_yaml'


def _spider(resource_path, ignore, followlinks, hidden, extra_kwargs):
    resources = {}
    for filename in (os.path.join(resource_path, x)
                     for x in os.listdir(resource_path)):
        basename = os.path.basename(filename)
        if (basename.startswith(os.curdir) and not hidden or
                os.path.islink(filename) and not followlinks):
            continue
        if os.path.isdir(filename):
            new_resources = _spider(filename, ignore=ignore,
                                    followlinks=followlinks,
                                    hidden=hidden,
                                    extra_kwargs=extra_kwargs)
            if new_resources:
                resources[basename] = new_resources
        else:
            with ignoring(*ignore):
                resources[basename] = bz_data(filename, **(extra_kwargs or {}))
    return resources


def data_spider(path,
                ignore=(ValueError, NotImplementedError),
                followlinks=True,
                hidden=False,
                extra_kwargs=None):
    """Traverse a directory and call ``blaze.data`` on its contents.

    Parameters
    ----------
    path : str
        Path to a directory of resources to load
    ignore : tuple of Exception, optional
        Ignore these exceptions when calling ``blaze.data``
    followlinks : bool, optional
        Follow symbolic links
    hidden : bool, optional
        Load hidden files
    extra_kwargs: dict, optional
        extra kwargs to forward on to ``blaze.data``

    Returns
    -------
    dict
        Possibly nested dictionary of containing basenames mapping to resources
    """
    # NOTE: this is named `data_spider` rather than just `spider` to
    # disambiguate this function from the `blaze.server.spider` module.
    return {os.path.basename(path): _spider(path,
                                            ignore=ignore,
                                            followlinks=followlinks,
                                            hidden=hidden,
                                            extra_kwargs=extra_kwargs)}


@contextmanager
def pushd(path):
    """Context manager that changes to ``path`` directory on enter and
    changes back to ``os.getcwd()`` on exit.
    """
    cwd = os.getcwd()
    os.chdir(os.path.abspath(path))
    try:
        yield
    finally:
        os.chdir(cwd)


def from_yaml(fh,
              ignore=(ValueError, NotImplementedError),
              followlinks=True,
              hidden=False,
              relative_to_yaml_dir=False):
    """Construct a dictionary of resources from a YAML specification.

    Parameters
    ----------
    fh : file
        File object referring to the YAML specification of resources to load.
    ignore : tuple of Exception, optional
        Ignore these exceptions when calling ``blaze.data``.
    followlinks : bool, optional
        Follow symbolic links.
    hidden : bool, optional
        Load hidden files.
    relative_to_yaml_dir: bool, optional, default False
        Load paths relative to yaml file's directory.  Default is to load
        relative to process' CWD.

    Returns
    -------
    dict
        A dictionary mapping top level keys in a YAML file to resources.

    See Also
    --------
    data_spider : Traverse a directory tree for resources
    """
    resources = {}
    yaml_dir = os.path.dirname(os.path.abspath(fh.name))
    for name, info in yaml.load(fh.read()).items():
        with pushd(yaml_dir if relative_to_yaml_dir else os.getcwd()):
            try:
                source = info.pop('source')
            except KeyError:
                raise ValueError('source key not found for data source named %r' %
                                 name)
            for mod in info.pop('imports', []):
                importlib.import_module(mod)
            if os.path.isdir(source):
                resources[name] = data_spider(os.path.expanduser(source),
                                              ignore=ignore,
                                              followlinks=followlinks,
                                              hidden=hidden,
                                              extra_kwargs=info)
            else:
                resources[name] = bz_data(source, **info)
    return resources


def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('path', type=argparse.FileType('r'), nargs='?',
                   default=None,
                   help='A YAML file specifying the resources to load')
    p.add_argument('-p', '--port', type=int, default=DEFAULT_PORT,
                   help='Port number')
    p.add_argument('-H', '--host', type=str, default='127.0.0.1',
                   help='Host name. Use 0.0.0.0 to listen on all public IPs')
    p.add_argument('-l', '--follow-links', action='store_true',
                   help='Follow links when listing files')
    p.add_argument('-e', '--ignored-exception', nargs='+',
                   default=['Exception'],
                   help='Exceptions to ignore when calling ``blaze.data`` on a file')
    p.add_argument('-d', '--hidden', action='store_true',
                   help='Call ``blaze.data`` on hidden files')
    p.add_argument('--yaml-dir', action='store_true',
                   help='Load path-based resources relative to yaml file directory.')
    p.add_argument('--allow-dynamic-addition', action='store_true',
                   help='Allow dynamically adding datasets to the server')
    p.add_argument('-D', '--debug', action='store_true',
                   help='Start the Flask server in debug mode')
    args = p.parse_args()
    if not (args.path or args.allow_dynamic_addition):
        msg = "No YAML file provided and --allow-dynamic-addition flag not set."
        p.error(msg)
    return args


def _main():
    args = _parse_args()
    ignore = tuple(getattr(builtins, e) for e in args.ignored_exception)
    if args.path:
        resources = from_yaml(args.path,
                              ignore=ignore,
                              followlinks=args.follow_links,
                              hidden=args.hidden,
                              relative_to_yaml_dir=args.yaml_dir)
    else:
        resources = {}
    server = Server(resources, allow_add=args.allow_dynamic_addition)
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    _main()
