from contextlib import contextmanager

# Don't clobber builtins
@contextmanager
def nobuiltins():
    gls = globals()
    builtins = gls.pop('__builtins__')
    yield
    gls['__builtins__'] = builtins
