import sys, os
import blaze
from blaze.io.server.app import app

if len(sys.argv) > 1:
    cat_path = sys.argv[1]
else:
    cat_path = os.path.join(os.getcwdu(), 'sample_arrays.yaml')

# Load the sample catalog, or from the selected path
blaze.catalog.load_config(cat_path)
print('Starting Blaze Server')
app.run(debug=True, port=8080, use_reloader=True)
