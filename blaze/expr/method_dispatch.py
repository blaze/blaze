
def match(condition, data):
    """ Does the condition match the data

    >>> match(1, 1)
    True
    >>> match(int, 1)
    True
    >>> match(lambda x: x > 0, 1)
    True

    Use tuples for many possibilities

    >>> match((1, 2, 3), 1)
    True
    >>> match((int, float), 1)
    True
    """
    return ((condition == data) or
            (isinstance(condition, type)
                and isinstance(data, condition)) or
            (not isinstance(condition, type)
                and callable(condition)
                and condition(data)) or
            (isinstance(condition, tuple)
                and any(match(c, data) for c in condition)))

def select_functions(methods, data):
    """
    Select appropriate functions given types and predicates
    """
    s = set()
    for condition, funcs in methods:
        if match(condition, data):
            s |= funcs
    return {func.__name__: func for func in s}
