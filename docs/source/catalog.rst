Catalog
=======

.. highlight:: python

The Blaze catalog is a way to organize and work with your data. In
the current version, it is implemented for a local catalog on one machine,
and in future revisions it will be extended to be consistent across
Blaze clusters.

Guided Tour
-----------

By default, when you start Blaze for the first time, an empty catalog
is created for you as a place to put your data. The default catalog
is configured at ``~/.blaze/catalog.config``, and it is set to point
to ``~/Arrays`` as the source of array data.

Loading Blaze in IPython
~~~~~~~~~~~~~~~~~~~~~~~~

The Blaze source repository includes a small set of sample arrays
in a catalog, we will load and work with these arrays to illustrate
the features. To start, load an IPython prompt, and import Blaze.::

    In [1]: import blaze

The first thing to do, since we want to demonstrate our sample catalog,
is to load that catalog's configuration.::

    In [4]: blaze.catalog.load_config(
       ...:     r"D:\Develop\blaze\samples\server\sample_arrays.yaml")

Navigating the Catalog
~~~~~~~~~~~~~~~~~~~~~~

On import, because we are in an IPython environment, Blaze has
configured some line magics to work with the catalog. These are
similar to the usual unix ``ls``, ``cd``, and ``pwd``, but prefixed
with ``b`` for Blaze. Let's start by taking a look what our working
directory is in the Blaze catalog, and what arrays and directories
are available. Note that you may have to say ``%bpwd`` instead of
``bpwd`` in some circumstances, but we will avoid the percent and
rely on IPython's magic.::

    In [5]: bpwd
    Out[5]: '/'

    In [6]: bls
    btc dates_vals kiva kiva_tiny

In the latter example, the directories should appear in a different
color than the arrays. You may see that ``btc``, ``kiva``, and
``kiva_tiny`` are directories, while ``dates_vals`` is an array.
Let's switch to the ``kiva_tiny`` directory.::

    In [8]: bcd kiva_tiny
    /kiva_tiny

    In [9]: bpwd
    Out[9]: u'/kiva_tiny'

    In [10]: bls
    lenders loans

Loading Array Data
~~~~~~~~~~~~~~~~~~

There are two arrays available here. Let's load the ``lenders`` and
take a look at its data. This is done with the ``%barr`` magic,
which can be used on the right side of an assignment statement.::

    In [11]: x = %barr lenders

    In [12]: x.dshape
    Out[12]: dshape("52, { lender_id : json; name : json; image : { id : int64; template_id : int64 }; whereabouts : string; country_code : json; uid : string; member_since : string; personal_url : string; occupation : string; loan_because : string; occupational_info : string; loan_count : int64; invitee_count : int64; inviter_id : json }")

Presently, the Blaze printing code has some difficulty with structures
of this complexity, so you may get an error if you try to print this.
[TODO: update this text when that is fixed.] We can, however, grab a few
of these fields, and print them using the underlying DyND to illustrate
the data.::

    In [15]: x.occupation._data.dynd_arr()
    Out[15]: nd.array(["International Development Manager", "retired dentist", "", "", "", "internet campaigner", "Engineer", "Nursery Manager", "Web designer", "Technical Assistant", "", "", "", "", "Licenses Optician", "physician", "", "Lic. Optician", "", "", "", "Master", "Politician", "guardian", "", "", "Carpenter", "", "IT", "Art Director", "Nomad", "Student", "Teacher", "IT Network consultant", "Pastry Chef", "Student", "", "Writer and Editor", "Union Construction", "veterinarian", "Architect", "Technical Trainer", "pharmacist / farmaceutico", "Marketing Manager", "teacher", "", "", "", "", "retired", "Sales Director", ""], strided_dim<string>)

The Array Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~

Let's now take a look at what the files on disk look like, backing
the array we loaded. The ``lenders`` array is described by a file ``lenders.array``,
as follows.::

    In [17]: print open(r"D:\Develop\blaze\samples\server\arrays\kiva_tiny\lenders.array").read()
    type: json
    import: {}
    datashape: |
        var, {
            lender_id: json; # Option(string);
            name: json; # Option(string);
            image: {
                id: int64;
                template_id: int64;
            };
            whereabouts: string;
            country_code: json; # string(2);
            uid: string;
            member_since: string; # datetime<minutes>
            personal_url: string; # URL type?
            occupation: string;
            loan_because: string;
            occupational_info: string;
            loan_count: int64;
            invitee_count: int64;
            inviter_id: json; # sometimes string, sometimes number 0. 0 is being used as the "missing value"
        }

This file is a YAML configuration file, which describes some data import
settings including the datashape of the data. In this case, the data is
in JSON format (``lenders.json`` right beside the ``.array`` file), and
its datashape is provided using YAML's multi-line string literal syntax.

If we go back up one level to the root of the catalog, we can see another
array ``dates_vals``, which is in CSV format. Here is how it looks,
once again using DyND to print the values.::

    In [18]: bcd ..
    /

    In [19]: x = %barr dates_vals

    In [20]: x._data.dynd_arr()
    Out[20]: nd.array([[2013-01-01, 0], [2013-01-02, 1], [2013-01-03, 2], [2013-01-04, 3], [2013-01-05, 4], [2013-01-06, 5], [2013-01-07, 6], [2013-01-08, 7], [2013-01-09, 8], [2013-01-10, 9], [2013-01-11, 10], [2013-01-12, 11], [2013-01-13, 12], [2013-01-14, 13], [2013-01-15, 14], [2013-01-16, 15], [2013-01-17, 16], [2013-01-18, 17], [2013-01-19, 18], [2013-01-20, 19], [2013-01-21, 20], [2013-01-22, 21], [2013-01-23, 22], [2013-01-24, 23], [2013-01-25, 24], [2013-01-26, 25], [2013-01-27, 26], [2013-01-28, 27], [2013-01-29, 28], [2013-01-30, 29], [2013-01-31, 30], [2013-02-01, 31], [2013-02-02, 32], [2013-02-03, 33], [2013-02-04, 34], [2013-02-05, 35], [2013-02-06, 36], [2013-02-07, 37], [2013-02-08, 38], [2013-02-09, 39], [2013-02-10, 40], [2013-02-11, 41], [2013-02-12, 42], [2013-02-13, 43], [2013-02-14, 44], [2013-02-15, 45], [2013-02-16, 46], [2013-02-17, 47], [2013-02-18, 48], [2013-02-19, 49], [2013-02-20, 50], [2013-02-21, 51], [2013-02-22, 52], [2013-02-23, 53], [2013-02-24, 54], [2013-02-25, 55], [2013-02-26, 56], [2013-02-27, 57], [2013-02-28, 58], [2013-03-01, 59], [2013-03-02, 60], [2013-03-03, 61], [2013-03-04, 62], [2013-03-05, 63], [2013-03-06, 64], [2013-03-07, 65], [2013-03-08, 66], [2013-03-09, 67], [2013-03-10, 68], [2013-03-11, 69], [2013-03-12, 70], [2013-03-13, 71], [2013-03-14, 72]], strided_dim<{data : date; values : int32}>)

