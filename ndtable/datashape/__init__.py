from parse import parse

# Emulate dtype('i') kind of behavior
def datashape(identifier):
    return parse(identifier)
