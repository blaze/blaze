"""
Errors raised by any SQL operation.
"""

from __future__ import absolute_import, division, print_function

from blaze import error

class sqlerror(error.BlazeException):
    """Base exception for SQL backend related errors"""
