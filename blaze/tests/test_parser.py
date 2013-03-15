from blaze.datashape import *
from blaze.datashape.parser import parse
from blaze.datashape.record import RecordDecl, derived
from blaze.datashape.coretypes import _reduce

from unittest import skip

def test_simple_parse():
    x = parse('2, 3, int32')
    y = parse('300 , 400, {x: int64; y: int32}')

    assert type(x) == DataShape
    assert type(y) == DataShape

    assert type(y[0]) == Fixed
    assert type(y[1]) == Fixed
    assert type(y[2]) == Record

    rec = y[2]

    assert rec['x'] == int64
    assert rec['y'] == int32

def test_compound_record1():
    p = parse('6, {x:int; y:float; z:str}')

    assert type(p[0]) == Fixed
    assert type(p[1]) == Record

def test_compound_record2():
    p = parse('{ a: { x: int; y: int }; b: {w: int; u: int } }')

    assert type(p) == Record

def test_free_variables():
    p = parse('N, M, 800, 600, int32')

    assert type(p[0]) == TypeVar
    assert type(p[1]) == TypeVar
    assert type(p[2]) == Fixed
    assert type(p[3]) == Fixed
    assert type(p[4]) == CType

def test_parse_equality():
    x = parse('800, 600, int64')
    y = parse('800, 600, int64')

    assert x._equal(y)

def test_parse_vars():
    x = parse('Range(1,2), int32')

    assert x[0].lower == 1
    assert x[0].upper == 2

def test_parse_either():
    x = parse('Either(int64, float64)')

    assert type(x) == Either
    assert x.a == int64
    assert x.b == float64

def test_custom_record():

    class Stock1(RecordDecl):
        name   = string
        open   = float_
        close  = float_
        max    = int64
        min    = int64
        volume = float_

        @derived('int64')
        def mid(self):
            return (self.min + self.max)/2

    assert Stock1.mid

def test_fields_with_reserved_names():
    # Should be able to name a field 'type', 'int64'
    # or any other word otherwise reserved in the
    # datashape language
    x = parse("""{
            type: bool;
            blob: bool;
            bool: bool;
            int: int32;
            float: float32;
            double: float64;
            int8: int8;
            int16: int16;
            int32: int32;
            int64: int64;
            uint8: uint8;
            uint16: uint16;
            uint32: uint32;
            uint64: uint64;
            float16: float32;
            float32: float32;
            float64: float64;
            float128: float64;
            complex64: float32;
            cfloat32: float32;
            complex128: float64;
            cfloat64: float64;
            string: string;
            object: string;
            datetime: string;
            datetime64: string;
            timedelta: string;
            timedelta64: string;
            json: string;
        }""")

def test_kiva_datashape():
    # A slightly more complicated datashape which should parse
    x = parse("""5, VarDim, {
          id: int64;
          name: string;
          description: {
            languages: VarDim, string(2);
            texts: json;
          };
          status: string;
          funded_amount: float64;
          basket_amount: json;
          paid_amount: json;
          image: {
            id: int64;
            template_id: int64;
          };
          video: json;
          activity: string;
          sector: string;
          use: string;
          delinquent: bool;
          location: {
            country_code: string(2);
            country: string;
            town: json;
            geo: {
              level: string;
              pairs: string;
              type: string;
            };
          };
          partner_id: int64;
          posted_date: json;
          planned_expiration_date: json;
          loan_amount: float64;
          currency_exchange_loss_amount: json;
          borrowers: VarDim, {
            first_name: string;
            last_name: string;
            gender: string(1);
            pictured: bool;
          };
          terms: {
            disbursal_date: json;
            disbursal_currency: string(3,'A');
            disbursal_amount: float64;
            loan_amount: float64;
            local_payments: VarDim, {
              due_date: json;
              amount: float64;
            };
            scheduled_payments: VarDim, {
              due_date: json;
              amount: float64;
            };
            loss_liability: {
              nonpayment: string;
              currency_exchange: string;
              currency_exchange_coverage_rate: json;
            };
          };
          payments: VarDim, {
            amount: float64;
            local_amount: float64;
            processed_date: json;
            settlement_date: json;
            rounded_local_amount: float64;
            currency_exchange_loss_amount: float64;
            payment_id: int64;
            comment: json;
          };
          funded_date: json;
          paid_date: json;
          journal_totals: {
            entries: int64;
            bulkEntries: int64;
          };
        }
    """)
