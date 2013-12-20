from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper
from ... import py2help

def crossdomain(origin=None, methods=None, headers=None,
                automatic_headers=True,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, py2help.basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, py2help.basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        return methods
        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if automatic_headers and h.get('Access-Control-Request-Headers'):
                h['Access-Control-Allow-Headers'] = h['Access-Control-Request-Headers']
            return resp
        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator
