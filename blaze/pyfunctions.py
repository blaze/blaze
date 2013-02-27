from ctypes import CFUNCTYPE, PYFUNCTYPE

class cfn(object):
    def __init__(self, name, argtys, retty):
        self.name = name
        self.argtys = argtys
        self.retty = retty

    def __str__(self):
        return 'foreign ccall %s :: %s -> %s' % \
            (self.name, self.argtys, self.retty)

def wrap_cfn(ctx, argtys, retty, address, gil=True):
    cargtys = [ctx.types[argty] for argty in argty]
    cretty = ctx.types[retty]

    if gil:
        ffi = PYFUNCTYPE
    else:
        ffi = CFUNCTYPE

    callable = ffi(cretty, *cargtys)(address)
    return callable
