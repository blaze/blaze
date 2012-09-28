# Test ArrayServer, generator-like memory access over tcp
# Supports publishing new arrays to the server. Is "threaded"
# with greenlets so whenever publishing to an exisiting array a
# lock is acquired on that array.

# A connection between the client and server over via a HTTP-like
# protocol.

# ----------------------
# array://localhost:5001
# ----------------------
#
# Client Request
# ==============

# GET /example\r\n
# Stride: 800\r\n
# Offset: 0\r\n
# \r\n

# Server Response
# ===============

# OK
# Connection: Keep-Alive
#
# \x00\x00\x00\x00\x00\x00\x00\x15
# \x0l\x00\x03\x00\x00\x00\x00\x13
# ....

import numpy as np
from numpy.random import random_integers

from gevent.coros import Semaphore
from gevent.server import StreamServer

from collections import defaultdict

na = random_integers(0, 100, (100,100))

locks = defaultdict(Semaphore)
store = {
    'example': na
}

def get(url, socket, stride, offset):
    ar = store[url]

    view = memoryview(ar[offset:])
    while view:
        nsent = socket.send(view, stride)
        view = view[nsent:]

def post(url, socket, datashape):
    if url in store:
        view = memoryview(store[url])
        ar = store[url]
        stride = ar.strides[0]
    else:
        shape, dtype = datashape
        ar = np.ndarray(shape, dtype)
        stride = ar.strides[0]

        store[url] = np.ndarray(shape, dtype)
        view = memoryview(store[url])

    while view:
        nbytes = socket.recv_into(view)
        view = view[stride:]

def application(socket, address):
    print 'Connection from', address

    fd = socket.makefile()
    header = fd.readline()
    stride = socket.readline()
    offset = socket.readline()
    method, url = header.split()

    if method == 'GET':
        with locks(url):
            get(url, socket, stride, offset)
    if method == 'POST':
        with locks(url):
            post(url, socket, stride, offset)

if __name__ == '__main__':
    server = StreamServer(('0.0.0.0', 5001), application)
    print ('Started ArrayServer on array://localhost:5001')
    server.serve_forever()
