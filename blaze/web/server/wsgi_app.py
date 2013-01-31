import sys, traceback
from dynd import nd, ndt
import json
from cgi import parse_qs
from blaze_url import split_array_base
from datashape_html import render_dynd_datashape

def wsgi_reconstruct_base_url(environ):
    from urllib import quote
    url = environ['wsgi.url_scheme']+'://'

    if environ.get('HTTP_HOST'):
        url += environ['HTTP_HOST']
    else:
        url += environ['SERVER_NAME']

        if environ['wsgi.url_scheme'] == 'https':
            if environ['SERVER_PORT'] != '443':
               url += ':' + environ['SERVER_PORT']
        else:
            if environ['SERVER_PORT'] != '80':
               url += ':' + environ['SERVER_PORT']

    url += quote(environ.get('SCRIPT_NAME', ''))
    return url

def slice_as_interior_string(s):
    if type(s) is int:
        return str(s)
    else:
        result = ''
        if s.start is not None:
            result += str(s.start)
        result += ':'
        if s.stop is not None:
            result += str(s.stop)
        if s.step is not None and s.step != 1:
            result += ':' + str(s.step)
        return result

def slice_as_string(s):
    return '[' + slice_as_interior_string(s) + ']'

def index_tuple_as_string(s):
    result = '[' + slice_as_interior_string(s[0])
    for i in s[1:]:
        result += ', ' + slice_as_interior_string(i)
    result += ']'
    return result

def add_indexers_to_url(base_url, indexers):
    for idx in indexers:
        if type(idx) is str:
            base_url += '.' + idx
        elif type(idx) is int:
            base_url += '[' + str(idx) + ']'
        elif type(idx) is slice:
            base_url += slice_as_string(idx)
    return base_url

def indexers_navigation_html(base_url, array_name, indexers):
    base_url = base_url + array_name
    result = '<a href="' + base_url + '">' + array_name + '</a>'
    for i, idx in enumerate(indexers):
        if type(idx) is str:
            base_url = base_url + '.' + idx
            result += (' . <a href="' + base_url + '">' + idx + '</a>')
        elif type(idx) is int:
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
        elif type(idx) is slice:
            s = slice_as_string(idx)
            base_url = base_url + s
            result += (' <a href="' + base_url + '">' + s + '</a>')
        elif type(idx) is tuple:
            s = index_tuple_as_string(idx)
            base_url = base_url + s
            result += (' <a href="' + base_url + '">' + s + '</a>')
    return result

class wsgi_app:
    def __init__(self, array_provider):
        self.array_provider = array_provider
 
    def __call__(self, environ, start_response):
        array_name, indexers = split_array_base(environ['PATH_INFO'])

        # Request the array from the array provider
        arr = self.array_provider(array_name)
        if arr is None:
            start_response('404 Not Found', [('content-type', 'text/plain')])
            return ['No Blaze Array named ' + array_name]

        print "Got request: " + environ['PATH_INFO'] + environ['QUERY_STRING']
        print "Array name: " + array_name
        print "Indexers: " + str(indexers)

        try:
            for i in indexers:
                if type(i) in [slice, int, tuple]:
                    arr = arr[i]
                else:
                    arr = getattr(arr, i)
        except:
            status = '400 Bad Request'
            response_headers = [('content-type', 'text/plain')]
            start_response(status, response_headers, sys.exc_info())
            return ['Bad indexer to Blaze Array\n\n' + traceback.format_exc()]

        if environ['QUERY_STRING'] == '' and environ['REQUEST_METHOD'] == 'GET':
            # This version of the array information is for human consumption
            content_type = 'text/html; charset=utf-8'
            base_url = wsgi_reconstruct_base_url(environ)
            array_url = base_url + environ.get('PATH_INFO', '')
            nav_html = indexers_navigation_html(base_url, array_name, indexers)
            datashape_html = render_dynd_datashape(array_url, arr)
            body = '<html><head><title>Blaze Array</title></head>\n' + \
                '<body>\n' + \
                'Blaze Array &gt; ' + nav_html + '\n<p />\n' + \
                '<a href="' + array_url + '?r=data.json">JSON</a>\n<p />\n' + \
                datashape_html + \
                '</body></html>'
        else:
            q = parse_qs(environ['QUERY_STRING'])
            print q
            if not q.has_key('r'):
                status = '400 Bad Request'
                response_headers = [('content-type', 'text/plain')]
                start_response(status, response_headers, sys.exc_info())
                return ['Blaze server request requires the ?r= query request type']
            if q['r'][0] == 'data.json':
                content_type = 'application/json; charset=utf-8'
                body = nd.format_json(arr).view_scalars(ndt.bytes).as_py()
            elif q['r'][0] == 'datashape':
                content_type = 'application/json; charset=utf-8'
                body = arr.dshape
            else:
                status = '400 Bad Request'
                response_headers = [('content-type', 'text/plain')]
                start_response(status, response_headers, sys.exc_info())
                return ['Unknown Blaze server request ?r=%s' % q['r'][0]]
        
        status = '200 OK'
        response_headers = [
            ('content-type', content_type),
            ('content-length', str(len(body)))
        ]
        start_response(status, response_headers)
        return [body]
