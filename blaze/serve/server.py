from flask import Flask

app = Flask('Blaze-Server')

datasets = dict()

@app.route('/')
def hello():
    return 'Welcome to Blaze Server'

