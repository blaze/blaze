import sys, os
from blaze.server.app import app
from blaze.catalog.array_provider import json_array_provider

if len(sys.argv) > 1:
    root_path = sys.argv[1]
else:
    root_path = os.path.join(os.getcwdu(), 'arrays')

array_provider = json_array_provider(root_path)
app.array_provider = array_provider
print('Starting Blaze Server')
app.run(debug=True, port=8080, use_reloader=True)
