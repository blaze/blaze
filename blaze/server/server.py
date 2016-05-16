from __future__ import absolute_import, division, print_function

import collections
from datetime import datetime
import errno
import functools
from hashlib import md5
import os
import socket
from time import time
from warnings import warn
import importlib

from datashape import discover, pprint
import flask
from flask import Blueprint, Flask, request, Response
from flask.ext.cors import cross_origin
from werkzeug.http import parse_options_header
from toolz import valmap

import blaze
from blaze import compute, resource
from blaze.compatibility import ExitStack
from blaze.compute import compute_up
from blaze.expr import utils as expr_utils

from .serialization import json, all_formats
from ..interactive import _Data
from ..expr import Expr, symbol


__all__ = 'Server', 'to_tree', 'from_tree', 'expr_md5'

# http://www.speedguide.net/port.php?port=6363
# http://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers
DEFAULT_PORT = 6363


class RC(object):
    """
    Simple namespace for HTTP status codes.
    https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
    """

    OK = 200
    CREATED = 201

    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    FORBIDDEN = 403
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    UNSUPPORTED_MEDIA_TYPE = 415

    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501


api = Blueprint('api', __name__)
pickle_extension_api = Blueprint('pickle_extension_api', __name__)


_no_default = object()  # sentinel


def _get_option(option, options, default=_no_default):
    try:
        return options[option]
    except KeyError:
        if default is not _no_default:
            return default

        # Provides a more informative error message.
        msg = 'The blaze api must be registered with {option}'
        raise TypeError(msg.format(option=option))


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _register_api(app, options, first_registration=False):
    """
    Register the data with the blueprint.
    """
    _get_data.cache[app] = _get_option('data', options)
    _get_format.cache[app] = {f.name: f for f in _get_option('formats', options)}
    _get_auth.cache[app] = (_get_option('authorization', options, None) or
                            (lambda a: True))
    allow_profiler = _get_option('allow_profiler', options, False)
    profiler_output = _get_option('profiler_output', options, None)
    profile_by_default = _get_option('profile_by_default', options, False)
    if not allow_profiler and (profiler_output or profile_by_default):
        msg = "cannot set %s%s%s when 'allow_profiler' is False"
        raise ValueError(msg % ('profiler_output' if profiler_output else '',
                                ' or ' if profiler_output and profile_by_default else '',
                                'profile_by_default' if profile_by_default else ''))
    if allow_profiler:
        if profiler_output is None:
            profiler_output = 'profiler_output'
        if profiler_output != ':response':
            ensure_dir(profiler_output)

    _get_profiler_info.cache[app] = (allow_profiler,
                                     profiler_output,
                                     profile_by_default)

    # Allowing users to dynamically add datasets to the Blaze server can be
    # dangerous, so we only expose the method if specifically requested
    allow_add = _get_option('allow_add', options, False)
    if allow_add:
        app.add_url_rule('/add', 'addserver', addserver,
                         methods=['POST', 'HEAD', 'OPTIONS'])

    # Call the original register function.
    Blueprint.register(api, app, options, first_registration)

api.register = _register_api


def per_app_accesor(name):
    def _get():
        return _get.cache[flask.current_app]
    _get.cache = {}
    _get.__name__ = '_get' + name
    return _get


def _get_format(name):
    return _get_format.cache[flask.current_app][name]
_get_format.cache = {}

_get_data = per_app_accesor('data')
_get_auth = per_app_accesor('auth')
_get_profiler_info = per_app_accesor('profiler_info')


def expr_md5(expr):
    """Returns the md5 hash of the str of the expression.

    Parameters
    ----------
    expr : Expr
        The expression to hash.

    Returns
    -------
    hexdigest : str
        The hexdigest of the md5 of the str of ``expr``.
    """
    exprstr = str(expr)
    if not isinstance(exprstr, bytes):
        exprstr = exprstr.encode('utf-8')
    return md5(exprstr).hexdigest()


def _prof_path(profiler_output, expr):
    """Get the path to write the data for a profile run of ``expr``.

    Parameters
    ----------
    profiler_output : str
        The director to write into.
    expr : Expr
        The expression that was run.

    Returns
    -------
    prof_path : str
        The filepath to write the new profiler data.

    Notes
    -----
    This function ensures that the dirname of the returned path exists.
    """
    dir_ = os.path.join(profiler_output,
                        expr_md5(expr))  # Use md5 so the client knows where to look.
    ensure_dir(dir_)
    return os.path.join(dir_,
                        str(int(datetime.utcnow().timestamp())))


def authorization(f):
    @functools.wraps(f)
    def authorized(*args, **kwargs):
        if not _get_auth()(request.authorization):
            return Response('bad auth token',
                            RC.UNAUTHORIZED,
                            {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return authorized


def check_request(f):
    @functools.wraps(f)
    def check():
        raw_content_type = request.headers['content-type']
        content_type, options = parse_options_header(raw_content_type)

        if content_type not in accepted_mimetypes:
            return ('Unsupported serialization format %s' % content_type,
                    RC.UNSUPPORTED_MEDIA_TYPE)

        try:
            serial = _get_format(accepted_mimetypes[content_type])
        except KeyError:
            return ("Unsupported serialization format '%s'" % matched.groups()[0],
                    RC.UNSUPPORTED_MEDIA_TYPE)

        try:
            payload = serial.loads(request.data)
        except ValueError:
            return ("Bad data.  Got %s " % request.data, RC.BAD_REQUEST)

        return f(payload, serial)
    return check


class Server(object):

    """ Blaze Data Server

    Host local data through a web API

    Parameters
    ----------
    data : dict, optional
        A dictionary mapping dataset name to any data format that blaze
        understands.
    formats : iterable, optional
        An iterable of supported serialization formats. By default, the
        server will support JSON.
        A serialization format is an object that supports:
        name, loads, and dumps.
    authorization : callable, optional
        A callable to be used to check the auth header from the client.
        This callable should accept a single argument that will either be
        None indicating that no header was passed, or an object
        containing a username and password attribute. By default, all requests
        are allowed.
    allow_profiler : bool, optional
        Allow payloads to specify `"profile": true` which will run the
        computation under cProfile.
    profiler_output : str, optional
        The directory to write pstats files after profile runs.
        The files will be written in a structure like:

          {profiler_output}/{hash(expr)}/{timestamp}

        This defaults to a relative path of `profiler_output`.
        This requires `allow_profiler=True`.

        If this is the string ':response' then writing to the local filesystem
        is disabled. Only requests that specify `profiler_output=':response'`
        will be served. All others will return a 403 (Forbidden).
    profile_by_default : bool, optional
        Run the profiler on any computation that does not explicitly set
        "profile": false.
        This requires `allow_profiler=True`.
    allow_add : bool, optional
        Expose an `/add` endpoint to allow datasets to be dynamically added to
        the server. Since this increases the risk of security holes, it defaults
        to `False`.

    Examples
    --------
    >>> from pandas import DataFrame
    >>> df = DataFrame([[1, 'Alice',   100],
    ...                 [2, 'Bob',    -200],
    ...                 [3, 'Alice',   300],
    ...                 [4, 'Dennis',  400],
    ...                 [5,  'Bob',   -500]],
    ...                columns=['id', 'name', 'amount'])

    >>> server = Server({'accounts': df})
    >>> server.run() # doctest: +SKIP
    """
    def __init__(self,
                 data=None,
                 formats=None,
                 authorization=None,
                 allow_profiler=False,
                 profiler_output=None,
                 profile_by_default=False,
                 allow_add=False):
        if isinstance(data, collections.Mapping):
            data = valmap(lambda v: v.data if isinstance(v, _Data) else v,
                          data)
        elif isinstance(data, _Data):
            data = data._resources()
        app = self.app = Flask('blaze.server.server')
        if data is None:
            data = {}
        app.register_blueprint(api,
                               data=data,
                               formats=formats if formats is not None else (json,),
                               authorization=authorization,
                               allow_profiler=allow_profiler,
                               profiler_output=profiler_output,
                               profile_by_default=profile_by_default,
                               allow_add=allow_add)
        self.data = data

    def run(self, port=DEFAULT_PORT, retry=False, **kwargs):
        """Run the server.

        Parameters
        ----------
        port : int, optional
            The port to bind to.
        retry : bool, optional
            If the port is busy, should we retry with the next available port?
        **kwargs
            Forwarded to the underlying flask app's ``run`` method.

        Notes
        -----
        This function blocks forever when successful.
        """
        self.port = port
        try:
            # Blocks until the server is shut down.
            self.app.run(port=port, **kwargs)
        except socket.error:
            if not retry:
                raise

            warn("Oops, couldn't connect on port %d.  Is it busy?" % port)
            # Attempt to start the server on a new port.
            self.run(port=port + 1, retry=retry, **kwargs)


@api.route('/datashape', methods=['GET'])
@cross_origin(origins='*', methods=['GET'])
@authorization
def shape():
    return pprint(discover(_get_data()), width=0)


def to_tree(expr, names=None):
    """ Represent Blaze expression with core data structures

    Transform a Blaze expression into a form using only strings, dicts, lists
    and base types (int, float, datetime, ....)  This form can be useful for
    serialization.

    Parameters
    ----------
    expr : Expr
        A Blaze expression

    Examples
    --------

    >>> t = symbol('t', 'var * {x: int32, y: int32}')
    >>> to_tree(t) # doctest: +SKIP
    {'op': 'Symbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}


    >>> to_tree(t.x.sum()) # doctest: +SKIP
    {'op': 'sum',
     'args': [{'op': 'Column',
               'args': [{'op': 'Symbol'
                         'args': ['t',
                                  'var * { x : int32, y : int32 }',
                                  False]}
                        'x']}]}

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``Symbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Column', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t.x

    See Also
    --------

    from_tree
    """
    if names and expr in names:
        return names[expr]
    if isinstance(expr, tuple):
        return [to_tree(arg, names=names) for arg in expr]
    if isinstance(expr, expr_utils._slice):
        return to_tree(expr.as_slice(), names=names)
    if isinstance(expr, slice):
        return {'op': 'slice',
                'args': [to_tree(arg, names=names) for arg in
                         [expr.start, expr.stop, expr.step]]}
    elif isinstance(expr, _Data):
        return to_tree(symbol(expr._name, expr.dshape), names)
    elif isinstance(expr, Expr):
        return {'op': type(expr).__name__,
                'args': [to_tree(arg, names) for arg in expr._args]}
    else:
        return expr


def expression_from_name(name):
    """

    >>> expression_from_name('By')
    <class 'blaze.expr.split_apply_combine.By'>

    >>> expression_from_name('And')
    <class 'blaze.expr.arithmetic.And'>
    """
    import blaze
    if hasattr(blaze, name):
        return getattr(blaze, name)
    if hasattr(blaze.expr, name):
        return getattr(blaze.expr, name)
    for signature, func in compute_up.funcs.items():
        try:
            if signature[0].__name__ == name:
                return signature[0]
        except TypeError:
            pass
    raise ValueError('%s not found in compute_up' % name)


def from_tree(expr, namespace=None):
    """ Convert core data structures to Blaze expression

    Core data structure representations created by ``to_tree`` are converted
    back into Blaze expressions.

    Parameters
    ----------
    expr : dict

    Examples
    --------

    >>> t = symbol('t', 'var * {x: int32, y: int32}')
    >>> tree = to_tree(t)
    >>> tree # doctest: +SKIP
    {'op': 'Symbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}

    >>> from_tree(tree)
    <`t` symbol; dshape='var * {x: int32, y: int32}'>

    >>> tree = to_tree(t.x.sum())
    >>> tree # doctest: +SKIP
    {'op': 'sum',
     'args': [{'op': 'Field',
               'args': [{'op': 'Symbol'
                         'args': ['t',
                                  'var * {x : int32, y : int32}',
                                  False]}
                        'x']}]}

    >>> from_tree(tree)
    sum(t.x)

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``Symbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Field', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t.x

    See Also
    --------

    to_tree
    """
    if isinstance(expr, dict):
        op, args = expr['op'], expr['args']
        if 'slice' == op:
            return expr_utils._slice(*[from_tree(arg, namespace)
                                       for arg in args])
        if hasattr(blaze.expr, op):
            cls = getattr(blaze.expr, op)
        else:
            cls = expression_from_name(op)
        if 'Symbol' in op:
            children = [from_tree(arg) for arg in args]
        else:
            children = [from_tree(arg, namespace) for arg in args]
        return cls(*children)
    elif isinstance(expr, (list, tuple)):
        return tuple(from_tree(arg, namespace) for arg in expr)
    if namespace and expr in namespace:
        return namespace[expr]
    else:
        return expr


accepted_mimetypes = {'application/vnd.blaze+{}'.format(x.name): x.name for x
                         in all_formats}


@api.route('/compute', methods=['POST', 'HEAD', 'OPTIONS'])
@cross_origin(origins='*', methods=['POST', 'HEAD', 'OPTIONS'])
@authorization
@check_request
def compserver(payload, serial):
    (allow_profiler,
     default_profiler_output,
     profile_by_default) = _get_profiler_info()
    requested_profiler_output = payload.get('profiler_output',
                                            default_profiler_output)
    profile = payload.get('profile')
    profiling = (allow_profiler and
                 (profile or (profile_by_default and requested_profiler_output)))
    if profile and not allow_profiler:
        return ('profiling is disabled on this server', RC.FORBIDDEN)

    with ExitStack() as response_construction_context_stack:
        if profiling:
            from cProfile import Profile

            if (default_profiler_output == ':response' and
                    requested_profiler_output != ':response'):
                # writing to the local filesystem is disabled
                return ("local filepaths are disabled on this server, only"
                        " ':response' is allowed for the 'profiler_output' field",
                        RC.FORBIDDEN)

            profiler_output = requested_profiler_output
            profiler = Profile()
            profiler.enable()
            # ensure that we stop profiling in the case of an exception
            response_construction_context_stack.callback(profiler.disable)

        @response_construction_context_stack.callback
        def log_time(start=time()):
            flask.current_app.logger.info('compute expr: %s\ntotal time (s): %.3f',
                                          expr,
                                          time() - start)

        ns = payload.get('namespace', {})
        compute_kwargs = payload.get('compute_kwargs') or {}
        odo_kwargs = payload.get('odo_kwargs') or {}
        dataset = _get_data()
        ns[':leaf'] = symbol('leaf', discover(dataset))

        expr = from_tree(payload['expr'], namespace=ns)
        assert len(expr._leaves()) == 1
        leaf = expr._leaves()[0]

        try:
            result = serial.materialize(compute(expr,
                                                {leaf: dataset},
                                                **compute_kwargs),
                                        expr.dshape,
                                        odo_kwargs)
        except NotImplementedError as e:
            return ("Computation not supported:\n%s" % e, RC.NOT_IMPLEMENTED)
        except Exception as e:
            return ("Computation failed with message:\n%s: %s" % (type(e).__name__, e),
                    RC.INTERNAL_SERVER_ERROR)

        response = {'datashape': pprint(expr.dshape, width=0),
                    'data': serial.data_dumps(result),
                    'names': expr.fields}

    if profiling:
        import marshal
        from pstats import Stats

        if profiler_output == ':response':
            from pandas.compat import BytesIO
            file = BytesIO()
        else:
            file = open(_prof_path(profiler_output, expr), 'wb')

        with file:
            # Use marshal to dump the stats data to the given file.
            # This is taken from cProfile which unfortunately does not have
            # an api that allows us to pass the file object directly, only
            # a file path.
            marshal.dump(Stats(profiler).stats, file)
            if profiler_output == ':response':
                response['profiler_output'] = {'__!bytes': file.getvalue()}

    return serial.dumps(response)


@cross_origin(origins='*', methods=['POST', 'HEAD', 'OPTIONS'])
@authorization
@check_request
def addserver(payload, serial):
    """Add a data resource to the server.

    The request should contain serialized MutableMapping (dictionary) like
    object, and the server should already be hosting a MutableMapping resource.
    """

    data = _get_data.cache[flask.current_app]

    if not isinstance(data, collections.MutableMapping):
        data_not_mm_msg = ("Cannot update blaze server data since its current "
                           "data is a %s and not a mutable mapping (dictionary "
                           "like).")
        return (data_not_mm_msg % type(data), RC.UNPROCESSABLE_ENTITY)

    if not isinstance(payload, collections.Mapping):
        payload_not_mm_msg = ("Need a dictionary-like payload; instead was "
                              "given %s of type %s.")
        return (payload_not_mm_msg % (payload, type(payload)),
                RC.UNPROCESSABLE_ENTITY)

    if len(payload) > 1:
        error_msg = "Given more than one resource to add: %s"
        return (error_msg % list(payload.keys()),
                RC.UNPROCESSABLE_ENTITY)

    [(name, resource_info)] = payload.items()

    if name in data:
        msg = "Cannot add dataset named %s, already exists on server."
        return (msg % name, RC.CONFLICT)

    try:
        imports = []
        if isinstance(resource_info, dict):
            # Extract resource creation arguments
            source = resource_info['source']
            imports = resource_info.get('imports', [])
            args = resource_info.get('args', [])
            kwargs = resource_info.get('kwargs', {})
        else:
            # Just a URI
            source, args, kwargs = resource_info, [], {}
        # If we've been given libraries to import, we need to do so
        # before we can create the resource.
        for mod in imports:
            importlib.import_module(mod)
        # Make a new resource and try to discover it.
        new_resource = {name: resource(source, *args, **kwargs)}
        # Discovery is a minimal consistency check to determine if the new
        # resource is valid.
        ds = discover(new_resource)
        if name not in ds.dict:
            raise ValueError("%s not added." % name)
    except NotImplementedError as e:
        error_msg = "Addition not supported:\n%s: %s"
        return (error_msg % (type(e).__name__, e),
                RC.UNPROCESSABLE_ENTITY)
    except Exception as e:
        error_msg = "Addition failed with message:\n%s: %s"
        return (error_msg % (type(e).__name__, e),
                RC.UNPROCESSABLE_ENTITY)
    else:
        # Now that we've established that the new resource is discoverable--and
        # thus exists and is accessible--we add the resource to the server.
        data.update(new_resource)

    return ('OK', RC.CREATED)
