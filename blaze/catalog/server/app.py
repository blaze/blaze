from __future__ import absolute_import, division, print_function

import sys
import os

import flask
from flask import request, Response

import blaze
import datashape
from dynd import nd, ndt
from blaze.catalog.array_provider import json_array_provider
from blaze.catalog.blaze_url import (split_array_base, add_indexers_to_url,
                                     slice_as_string, index_tuple_as_string)
from blaze.py2help import _inttypes, _strtypes

from .datashape_html import render_datashape
from .compute_session import compute_session
from .crossdomain import crossdomain


app = flask.Flask('blaze.server')
app.sessions = {}


def indexers_navigation_html(base_url, array_name, indexers):
    base_url = base_url + array_name
    result = '<a href="' + base_url + '">' + array_name + '</a>'
    for i, idx in enumerate(indexers):
        if isinstance(idx, _strtypes):
            base_url = base_url + '.' + idx
            result += (' . <a href="' + base_url + '">' + idx + '</a>')
        elif isinstance(idx, _inttypes):
            new_base_url = base_url + '[' + str(idx) + ']'
            result += (' <a href="' + new_base_url + '">[' + str(idx) + ']</a>')
            # Links to increment/decrement this indexer
            #result += '<font style="size:7px"><table cellpadding="0" cellspacing="0" border="0">'
            #result += '<tr><td><a href="'
            #result += add_indexers_to_url(base_url, [idx + 1] + indexers[i+1:])
            #result += '">/\\</a></td></tr>'
            #result += '<tr><td><a href="'
            #result += add_indexers_to_url(base_url, [idx - 1] + indexers[i+1:])
            #result += '">\\/</a></td></tr>'
            #result += '</table></font>'
            base_url = new_base_url
        elif isinstance(idx, slice):
            s = slice_as_string(idx)
            base_url = base_url + s
            result += (' <a href="' + base_url + '">' + s + '</a>')
        elif isinstance(idx, tuple):
            s = index_tuple_as_string(idx)
            base_url = base_url + s
            result += (' <a href="' + base_url + '">' + s + '</a>')
        else:
            raise IndexError('Invalid indexer %r' % idx)
    return result


def get_array(array_name, indexers):
    arr = blaze.catalog.get(array_name)
    for i in indexers:
        if type(i) in [slice, int, tuple]:
            arr = arr[i]
        else:
            ds = arr.dshape
            if isinstance(ds, datashape.DataShape):
                ds = ds[-1]
            if isinstance(ds, datashape.Record) and i in ds.names:
                arr = getattr(arr, i)
            else:
                raise Exception('Blaze array does not have field ' + i)
    return arr


def html_array(arr, base_url, array_name, indexers):
    array_url = add_indexers_to_url(base_url + array_name, indexers)
    print(array_url)

    nav_html = indexers_navigation_html(base_url, array_name, indexers)
    datashape_html = render_datashape(array_url, arr.dshape)
    body = '<html><head><title>Blaze Array</title></head>\n' + \
        '<body>\n' + \
        'Blaze Array &gt; ' + nav_html + '\n<p />\n' + \
        '<a href="' + array_url + '?r=data.json">JSON</a>\n<p />\n' + \
        datashape_html + \
        '\n</body></html>'
    return body


@app.route("/favicon.ico")
def favicon():
    return 'no icon'


@app.route("/<path:path>", methods=['GET', 'POST', 'OPTIONS'])
@crossdomain(origin="*", automatic_options=False, automatic_headers=True)
def handle(path):
    if request.path in app.sessions:
        return handle_session_query()
    else:
        return handle_array_query()


def handle_session_query():
    session = app.sessions[request.path]
    q_req = request.values['r']
    if q_req == 'close_session':
        content_type, body = session.close()
        return Response(body, mimetype='application/json')
    elif q_req == 'add_computed_fields':
        j = request.values['json']
        content_type, body = session.add_computed_fields(j)
        return Response(body, mimetype='application/json')
    elif q_req == 'sort':
        j = request.values['json']
        content_type, body = session.sort(j)
        return Response(body, mimetype='application/json')
    elif q_req == 'groupby':
        j = request.values['json']
        content_type, body = session.groupby(j)
        return Response(body, mimetype='application/json')
    else:
        return 'something with session ' + session.session_name


def handle_array_query():
    array_name, indexers = split_array_base(request.path.rstrip('/'))
    arr = get_array(array_name, indexers)
    base_url = request.url_root[:-1]
    #no query params
    # NOTE: len(request.values) was failing within werkzeug
    if len(list(request.values)) == 0:
        return html_array(arr, base_url, array_name, indexers)
    q_req = request.values['r']
    if q_req == 'data.json':
        dat = arr.ddesc.dynd_arr()
        return Response(nd.as_py(nd.format_json(dat).view_scalars(ndt.bytes)),
                        mimetype='application/json')
    elif q_req == 'datashape':
        content_type = 'text/plain; charset=utf-8'
        return str(arr.dshape)
    elif q_req == 'dyndtype':
        content_type = 'application/json; charset=utf-8'
        body = str(arr.dtype)
        return Response(body, mimetype='application/json')
    elif q_req == 'dynddebug':
        return arr.debug_repr()
    elif q_req == 'create_session':
        session = compute_session(base_url,
                                  add_indexers_to_url(array_name, indexers))
        app.sessions[session.session_name] = session
        content_type, body = session.creation_response()
        return Response(body, mimetype='application/json')
    else:
        abort(400, "Unknown Blaze server request %s" % q_req)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = os.path.join(os.getcwdu(), 'arrays')
    array_provider = json_array_provider(root_path)
    app.array_provider = array_provider
    app.run(debug=True, port=8080, use_reloader=True)
