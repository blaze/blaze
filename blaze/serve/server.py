from flask import Flask

app = Flask('Blaze-Server')

datasets = dict()

@app.route('/')
def hello():
    return 'Welcome to Blaze Server'

@app.route('/datasets.json')
def dataset():
    return '\n'.join(': '.join((name, str(dset.dshape))) for name, dset in
                                datasets.items())
