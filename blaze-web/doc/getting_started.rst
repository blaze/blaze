.. contents:: Table of Contents

****************************************
Getting Started Running the Blaze Server
****************************************

The Blaze Server is a component of Blaze which is able to serve
array data in a fashion similar to OPeNDAP, and provides the
communication protocol for Blaze clustered computation.

The current implementation is a work in progress, but can already
serve arrays and provide for some basic manipulations.

The current version of the server is implemented directly on
DyND, a low level dynamic array library being developed for
backend support of Blaze features. Over time, the server will
transition to using the higher-level Blaze interfaces, and DyND
will provide backend support for those interfaces.

Prerequisites
=============

DyND - https://github.com/ContinuumIO/dynd-python

    There is not currently a conda package for DyND, so presently it
    must be built from source.

blaze-web

    This is the package which contains the getting started guide you're
    presently reading. Run 'python setup.py install' from a cloned
    copy of the blaze-web repository to install it. It installs itself
    as blaze_web.

Running The Sample Server
=========================

Starting The Server
-------------------

In the 'example' directory, there is a script that starts Python's
wsgiref server to serve some sample array data. With both DyND and
blaze-web installed, you can start it by running::

    ~/Develop/blaze-web/example $ python start_server.py 
    Starting Blaze Server...

Viewing An Array
----------------

Now, in a web browser, navigate to http://localhost:8080/kiva_tiny/loans.
You should see something which starts like this::

    Blaze Array > /kiva_tiny/loans

    JSON

    type BlazeDataShape = 5, { # JSON
      header: { # JSON
        total: int64; # JSON
        page: int64; # JSON
        date: string; # JSON
        page_size: int64; # JSON
      };
      loans: VarDim, { # JSON
        id: int64; # JSON

This is looking at the files in the 'example/arrays/kiva_tiny/loans'
subdirectory, using the file 'example/arrays/kiva_tiny/loans.datashape'
for how to interpret the data in the JSON files.

Main Array Page
---------------

Click on the 'header' link (http://localhost:8080/kiva_tiny/loans.header),
to get to a simpler subarray we can use to illustrate the parts that are
visible. This page should look like this::

    Blaze Array > /kiva_tiny/loans . header

    JSON

    type BlazeDataShape = 5, { # JSON
      total: int64; # JSON
      page: int64; # JSON
      date: string; # JSON
      page_size: int64; # JSON
    }

    Debug Links: DyND Type   DyND Debug Repr 

Along the top is a navigation helper, which splits apart the indexers
used to get to the particular subarray being viewed. If you click on
the first link, labeled '/kiva_tiny/loans', you will get back to the original
array.

The link immediately below, labeled 'JSON', gives you the full data of the
array in JSON format. For this subarray, that gives you an array of the
five headers from the five JSON files making up the array.

Next is a rendition of the Blaze datashape for this array. If you click
on the 'BlazeDataShape' link, you will get the datashape as plaintext.

If you click on the link for any field name, it will take you to the page
for that field name, and if you click on the 'JSON' link beside a field
name, you will immediately get the JSON data you would get if you first
clicked on the field, then clicked on the JSON link at the top of the
resulting page.

Finally, at the bottom, are some links to help with debugging. They
give you some information about the DyND representation of the array
being viewed.

Indexing
--------

To pick specific elements of the array, you can add [] like in Python
to index into it. For example, to look at just one header, you
can go to `<http://localhost:8080/kiva_tiny/loans.header[2]>`__

Conclusion
==========

That's an overview of the Blaze array server. It's still a work in progress,
but hopefully this provides a small glimpse into the possibilities.

