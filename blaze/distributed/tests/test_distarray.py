"""Tests for the Distributed Array class and module"""

from __future__ import absolute_import, division, print_function

from blaze.distributed import DistArray

def test_dist_array_creation():
    """Simple test to create a distributed array"""
    DistArray(None)
