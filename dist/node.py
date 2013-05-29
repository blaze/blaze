import os
import sys
import uuid
import signal
import logging

import gevent
import zmq.green as zmq

import protocol as proto
from daemon import daemonize
from utils import next_available_port, zmq_addr

#------------------------------------------------------------------------
# Array Zerocopy
#------------------------------------------------------------------------

def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

#------------------------------------------------------------------------
# Storage Node
#------------------------------------------------------------------------

class StorageNode(object):

    def __init__(self):
        self.ctx = zmq.Context()

        self.started = False
        self.registered = False
        self.store = dict() # FIXME

        self.worker_id = str(uuid.uuid4())

        logging.basicConfig(logging=logging.DEBUG)
        logging.getLogger("").setLevel(logging.INFO)
        self.logging = logging

    def connect(self, store_addr = None,
                      push_addr  = None,
                      ctrl_addr  = None):

        c = zmq.Context()

        # -----------------------

        if not store_addr:
            self.store_addr = zmq_addr(8888, transport='tcp')
        else:
            self.store_addr = store_addr

        self.store_socket = c.socket(zmq.PULL)
        self.store_socket.connect(self.store_addr)

        # -----------------------

        if not push_addr:
            addr = zmq_addr(6666, transport='tcp')

        self.push_socket = c.socket(zmq.PUSH)
        self.push_socket.connect(addr)

        if not ctrl_addr:
            ctrl_addr = zmq_addr(7777, transport='tcp')

        # -----------------------

        self.ctrl_socket = c.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt(zmq.IDENTITY, self.worker_id)
        self.ctrl_socket.connect(ctrl_addr)

    def heartbeat_loop(self):
        while True:
            # FIXME
            gevent.sleep(1)

    def main_loop(self):
        self.started = True

        poller = zmq.Poller()
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.store_socket, zmq.POLLIN | zmq.POLLERR)

        while self.started:

            if self.registered:

                try:
                    events = dict(poller.poll())
                except zmq.ZMQError:
                    self._kill()
                    break

                if any(ev & zmq.POLLERR for ev in events.values()):
                    self.logging.error('Socket error')
                    self._kill()
                    break

                if events.get(self.ctrl_socket) == zmq.POLLIN:
                    worker_id, command = self.ctrl_socket.recv_multipart()
                    if command == proto.PING:
                        print 'pong'
                        self.ctrl_socket.send(proto.PONG)

            else:
                # Associate with the server
                self.push_socket.send_multipart([proto.CONNECT_STORE, self.worker_id, self.store_addr])
                worker_id, payload = self.ctrl_socket.recv_multipart()
                if payload == proto.PING:
                    self.registered = True

    def start(self, timeout=None):
        gevent.signal(signal.SIGQUIT, gevent.shutdown)
        self.logging.info('Started keyvalue node (%s)' % self.store_addr)

        # spawn two green threads
        main      = gevent.spawn(self.main_loop)
        heartbeat = gevent.spawn(self.heartbeat_loop)

        gevent.joinall([
            main,
            heartbeat,
        ])

    def _kill(self):
        self.socket.close()
        self.ctx.term()

    def handle(self, data):
        properties = data.split(":", 3)
        cmd = properties[0]

        key   = properties[1] if len(properties) > 1 else None
        value = properties[2] if len(properties) > 2 else None

        if cmd == proto.LIST:
            self.send(proto.ACK_LIST, self.store.keys())

        elif cmd == proto.SET:
            v = self.store[key] = value
            self.send(proto.ACK_SET, v)

        elif cmd == proto.GET:
            if key in self.store:
                value = self.store[key]
            else:
                value = ""
            self.send(proto.ACK_GET, value)

        elif cmd == proto.RENAME:
            v = self.store[key] = self.store[value]
            self.send(proto.ACK_RENAME, v)

        elif cmd == proto.DELETE:
            if key in self.store:
                value = self.store.pop(key)
            else:
                value = ""
            self.send(proto.ACK_DELETE, value)

        else:
            raise NotImplementedError

    def send(self, key, value):
        return self.push_socket.send("%s:%s" % (key.upper(), value), zmq.NOBLOCK)

def main():
    node = StorageNode()
    node.connect()
    node.start()

if __name__ == '__main__':
    main()
