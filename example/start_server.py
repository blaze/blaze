import sys, os
from os import path
from wsgiref.simple_server import make_server
from blaze.web.server.wsgi_app import wsgi_app
from blaze.web.server.array_provider import json_array_provider

if len(sys.argv) > 1:
    root_path = sys.argv[1]
else:
    root_path = os.path.join(os.getcwdu(), 'arrays')

app = wsgi_app(json_array_provider(root_path))
server = make_server('localhost', 8080, app)
server.serve_forever()
