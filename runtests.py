#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import blaze.testing

sys.exit(0 if blaze.testing.test(sys.argv[1:]) == 0 else 1)
