"""
Starts a Blaze server for tests.

$ start_test_server.py /path/to/catalog_config.yaml <portnumber>
"""
import sys, os
import blaze
from blaze.io.server.app import app

blaze.catalog.load_config(sys.argv[1])
app.run(port=int(sys.argv[2]), use_reloader=False)
