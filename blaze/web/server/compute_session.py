import json

class compute_session:
    def __init__(self, array_provider, base_url, array_name):
        self.array_provider = array_provider
        session_name, root_dir = array_provider.create_session_dir()
        self.session_name = array_name + '/' + session_name
        self.root_dir = root_dir
        self.array_name = array_name
        self.base_url = base_url
        
    def creation_response(self):
        content_type = 'application/json; charset=utf-8'
        body = json.dumps({
                'session' : self.base_url + self.session_name,
                'version' : 'prototype',
                'access' : 'no permission model yet'
            })
        return (content_type, body)