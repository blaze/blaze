import sys, traceback
from dynd import nd, ndt
from cgi import parse_qs
from blaze_url import split_array_base
from datashape_html import render_dynd_datashape
from compute_session import compute_session

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
        result += ',' + slice_as_interior_string(i)
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
        elif type(idx) is tuple:
            base_url += index_tuple_as_string(idx)
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
        # object which provides arrays from disk/other source
        self.array_provider = array_provider
        # Dictionary of open compute sessions
        self.sessions = {}

    def get_array(self, array_name, indexers):
        # Request the array from the array provider
        arr = self.array_provider(array_name)
        if arr is None:
            raise Exception('No Blaze Array named ' + array_name)

        for i in indexers:
            if type(i) in [slice, int, tuple]:
                arr = arr[i]
            else:
                arr = getattr(arr, i)
        return arr

    def html_array(self, arr, base_url, array_name, indexers):
        array_url = add_indexers_to_url(base_url + array_name, indexers)
        print array_url
        nav_html = indexers_navigation_html(base_url, array_name, indexers)
        datashape_html = render_dynd_datashape(array_url, arr)
        body = '<html><head><title>Blaze Array</title></head>\n' + \
            '<body>\n' + \
            'Blaze Array &gt; ' + nav_html + '\n<p />\n' + \
            '<a href="' + array_url + '?r=data.json">JSON</a>\n<p />\n' + \
            datashape_html + \
            '\n<p /> Debug Links: ' + \
            '<a href="' + array_url + '?r=dyndtype">DyND Type</a>\n' + \
            '&nbsp;&nbsp;' + \
            '<a href="' + array_url + '?r=dynddebug">DyND Debug Repr</a>\n' + \
            '</body></html>'
        return body

    def handle_session_query(self, environ, start_response):
        session = self.sessions[environ['PATH_INFO']]
        request_method = environ['REQUEST_METHOD']
        if request_method != 'POST':
            status = '404 Not Found'
            response_headers = [('content-type', 'text/plain')]
            start_response(status, response_headers, sys.exc_info())
            return ['Must use POST with compute session URL']
        else:
            # the environment variable CONTENT_LENGTH may be empty or missing
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0
            request_body = environ['wsgi.input'].read(request_body_size)
            q = parse_qs(request_body)

        print q
        if not q.has_key('r'):
            status = '400 Bad Request'
            response_headers = [('content-type', 'text/plain')]
            start_response(status, response_headers, sys.exc_info())
            return ['Blaze server compute session request requires the ?r= query request type']
        q_req = q['r'][0]
        if q_req == 'create_table_view':
            j = q['json'][0]
            content_type, body = session.create_table_view(j)

        content_type = 'text/plain; charset=utf-8'
        body = 'something with session ' + session.session_name

        status = '200 OK'
        response_headers = [
            ('content-type', content_type),
            ('content-length', str(len(body)))
        ]
        start_response(status, response_headers)
        return [body]
    
    def handle_array_query(self, environ, start_response):
        try:
            array_name, indexers = split_array_base(environ['PATH_INFO'])
            arr = self.get_array(array_name, indexers)

            base_url = wsgi_reconstruct_base_url(environ)
            request_method = environ['REQUEST_METHOD']
            if request_method == 'GET' and environ['QUERY_STRING'] == '':
                # This version of the array information is for human consumption
                content_type = 'text/html; charset=utf-8'
                body = self.html_array(arr, base_url, array_name, indexers)
            else:
                if request_method == 'GET':
                    q = parse_qs(environ['QUERY_STRING'])
                elif request_method == 'POST':
                    # the environment variable CONTENT_LENGTH may be empty or missing
                    try:
                        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                    except (ValueError):
                        request_body_size = 0
                    request_body = environ['wsgi.input'].read(request_body_size)
                    q = parse_qs(request_body)
                else:
                    status = '404 Not Found'
                    response_headers = [('content-type', 'text/plain')]
                    start_response(status, response_headers)
                    return ['Unsupported request method']
    
                print q
                if not q.has_key('r'):
                    status = '400 Bad Request'
                    response_headers = [('content-type', 'text/plain')]
                    start_response(status, response_headers, sys.exc_info())
                    return ['Blaze server request requires the ?r= query request type']
                q_req = q['r'][0]
                if q_req == 'data.json':
                    content_type = 'application/json; charset=utf-8'
                    body = nd.as_py(nd.format_json(arr).view_scalars(ndt.bytes))
                elif q_req == 'datashape':
                    content_type = 'text/plain; charset=utf-8'
                    body = 'type BlazeDataShape = ' + arr.dshape
                elif q_req == 'dyndtype':
                    content_type = 'application/json; charset=utf-8'
                    body = str(arr.dtype)
                elif q_req == 'dynddebug':
                    content_type = 'text/plain; charset=utf-8'
                    body = arr.debug_repr()
                elif q_req == 'create_session':
                    session = compute_session(self.array_provider, base_url,
                                              add_indexers_to_url(array_name, indexers))
                    self.sessions[session.session_name] = session
                    content_type, body = session.creation_response()
                else:
                    status = '400 Bad Request'
                    response_headers = [('content-type', 'text/plain')]
                    start_response(status, response_headers, sys.exc_info())
                    return ['Unknown Blaze server request ?r=%s' % q['r'][0]]
        except:
            status = '404 Not Found'
            response_headers = [('content-type', 'text/plain')]
            start_response(status, response_headers, sys.exc_info())
            return ['Error getting Blaze Array\n\n' + traceback.format_exc()]

        status = '200 OK'
        response_headers = [
            ('content-type', content_type),
            ('content-length', str(len(body)))
        ]
        start_response(status, response_headers)
        return [body]

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'] in self.sessions:
            return self.handle_session_query(environ, start_response)
        else:
            return self.handle_array_query(environ, start_response)
