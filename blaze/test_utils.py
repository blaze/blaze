class assert_raises(object):

    def __init__(self, exc_ty):
        self.exc_ty = exc_ty

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if type is None:
            raise AssertionError('Exception not raised')
        if issubclass(type, self.exc_ty):
            return True
        raise AssertionError('Wrong exception raised'), value
