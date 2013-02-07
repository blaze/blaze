

class compute_session:
    def __init__(self, array_provider):
        self.array_provider = array_provider
        self.session_name, self.root_dir = self.array_provider.create_session_dir()