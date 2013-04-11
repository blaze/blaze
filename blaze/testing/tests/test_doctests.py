# -*- coding: utf-8 -*-

"""
Test doctest support.
"""

def call_me(arg):
    """
    >>> call_me(5)
    """
    print arg

if __name__ == '__main__':
    import doctest
    try:
        doctest.testmod(raise_on_error=True)
    except doctest.DocTestFailure:
        pass
    else:
        raise Exception("Expected exception!")