
def select_functions(methods, data):
    """
    Select appropriate functions given types and predicates
    """
    s = set()
    for condition, funcs in methods:
        if isinstance(condition, (type, tuple)):
            if isinstance(data, condition):
                s |= funcs
        elif callable(condition) and condition(data):
            s |= funcs
    return {func.__name__: func for func in s}
