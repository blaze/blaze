"""
Errors raised by any scidb operation.
"""

from __future__ import absolute_import, division, print_function

from blaze import error

class scidberror(error.BlazeException):
    """Base exception for scidb backend related errors"""

class SciDBError(scidberror):
    """Raised when scidb complains about something."""

class InterfaceError(scidberror):
    """Raised when performing a query over different scidb interface handles"""