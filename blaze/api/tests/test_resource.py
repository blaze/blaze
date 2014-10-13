from blaze import resource, into
from blaze.data import Excel
import os
import pytest


dirname = os.path.dirname(__file__)

def test_into_xls_file():
    pytest.importorskip('xlrd')
    fn = os.path.join(dirname, 'accounts.xls')
    assert isinstance(resource(fn), Excel)


def test_into_xlsx_file():
    pytest.importorskip('xlrd')
    fn = os.path.join(dirname, 'accounts.xlsx')
    assert isinstance(resource(fn), Excel)


def test_into_directory_of_xlsx_files():
    pytest.importorskip('xlrd')
    fns = os.path.join(dirname, 'accounts_*.xlsx')
    assert into(list, fns) == [(1, 'Alice', 100),
                               (2, 'Bob', 200),
                               (3, 'Charlie', 300),
                               (4, 'Dan', 400),
                               (5, 'Edith', 500)]
