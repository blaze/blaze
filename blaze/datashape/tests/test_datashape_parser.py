import unittest

import blaze
from blaze import datashape
from blaze.datashape.parser import parse
from blaze.datashape.coretypes import Enum, Option

class TestDatashapeParser(unittest.TestCase):

    def test_simple_parse(self):
        x = parse('2, 3, int32')
        y = parse('300 , 400, {x: int64; y: int32}')

        assert type(x) == datashape.DataShape
        assert type(y) == datashape.DataShape

        assert type(y[0]) == datashape.Fixed
        assert type(y[1]) == datashape.Fixed
        assert type(y[2]) == datashape.Record

        rec = y[-1]

        assert rec.fields['x'] == datashape.int64
        assert rec.fields['y'] == datashape.int32

    def test_compound_record1(self):
        p = parse('6, {x:int32; y:float64; z:string}')

        assert type(p[0]) == datashape.Fixed
        assert type(p[1]) == datashape.Record

    def test_compound_record2(self):
        p = parse('{ a: { x: int32; y: int32 }; b: {w: int32; u: int32 } }')

        assert type(p) == datashape.Record

    def test_free_variables(self):
        p = parse('N, M, 800, 600, int32')

        assert type(p[0]) == datashape.TypeVar
        assert type(p[1]) == datashape.TypeVar
        assert type(p[2]) == datashape.Fixed
        assert type(p[3]) == datashape.Fixed
        assert type(p[4]) == datashape.CType

    def test_parse_equality(self):
        x = parse('800, 600, int64')
        y = parse('800, 600, int64')

        assert x._equal(y)

    def test_parse_vars(self):
        x = parse('Range(1,2), int32')

        assert x[0].lower == 1
        assert x[0].upper == 2

    def test_fields_with_reserved_names(self):
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
                var: string;
            }""")

    def test_kiva_datashape(self):
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


class TestCategorical(unittest.TestCase):

    def test_categorical_single(self):
        res = blaze.dshape('Categorical(Foo)')

        assert isinstance(res, Enum)

    def test_categorical_multi(self):
        res = blaze.dshape('Categorical(Foo, Bar)')

        assert isinstance(res, Enum)

    def test_categorical_module(self):
        source = """
        data Day = Monday | Tuesday | Wednesday | Thursday | Friday

        type Person = {
            days : Day;
        }
        """
        res = blaze.dshape(source, multi=True)

        assert isinstance(res[0], Enum)

class TestOption(unittest.TestCase):

    def test_categorical_single(self):
        res = blaze.dshape('Option(int32)')

        assert isinstance(res, Option)

    def test_categorical_multi(self):
        res = blaze.dshape('2, 3, Option(int32)')

        assert isinstance(res[2], Option)
