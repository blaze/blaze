# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

class MockedConn(object):

    def __init__(self):
        self.recorded = []

    def query(self, s):
        self.recorded.append(s)

    def wrap(self, arrname):
        raise NotImplementedError("Referencing remote scidb arrays")