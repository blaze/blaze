import imp
import signal
import marshal
import logging
import argparse
from types import ModuleType
from StringIO import StringIO
from collections import defaultdict

import gevent
import zmq.green as zmq

import configuration
import protocol as proto

from hashing import Ring
from utils import zmq_addr

try:
    import msgpack as srl
except ImportError:
    import cPickle as srl

try:
    import marshal as marshal
    #import dill as marshal
except ImportError:
    import marshal

# States
# ------
START     = 0
MAP       = 1
SHUFFLE   = 2
PARTITION = 3
REDUCE    = 4
COLLECT   = 5

# Shufle Backends
# ---------------
MEMORY = 0
DKV    = 1

class Server(object):

    def __init__(self, config, module, backend=MEMORY):
        self.state = START

        self.workers = set()
        self.stores = set()
        self.ring = Ring([], replicas=3)
        self.backend = backend
        self.config = config

        try:
            self.mapfn    = module.mapfn
            self.reducefn = module.reducefn
            self.datafn   = module.datafn
        except ImportError:
            raise RuntimeError('Given module does not expose interface')

        self.bytecode = None

        self.started = False
        self.completed = False

        self.working_maps = {}

        logging.basicConfig(logging=logging.DEBUG)
        logging.getLogger("").setLevel(logging.INFO)
        self.logging = logging

    def heartbeat_loop(self):
        while True:
            for worker in self.workers:
                print 'ping'
                self.ctrl_socket.send_multipart([worker, proto.PING])

            for store in self.stores:
                print 'ping'
                self.ctrl_socket.send_multipart([store, proto.PING])

            gevent.sleep(1)

    def main_loop(self):
        self.started = True

        poller = zmq.Poller()

        poller.register(self.pull_socket, zmq.POLLIN  | zmq.POLLERR)
        poller.register(self.push_socket, zmq.POLLOUT | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLOUT | zmq.POLLERR)
        poller.register(self.store_socket, zmq.POLLOUT | zmq.POLLERR)

        while self.started and not self.completed:
            try:
                events = dict(poller.poll())
            except zmq.ZMQError:
                self._kill()
                break

            if any(ev & zmq.POLLERR for ev in events.values()):
                self.logging.error('Socket error')
                self._kill()
                break

            # TODO: Specify number of nodes
            if len(self.workers) > 0:
                if events.get(self.push_socket) == zmq.POLLOUT:
                    self.start_new_task()
                if events.get(self.ctrl_socket) == zmq.POLLIN:
                    self.manage()
                if events.get(self.pull_socket) == zmq.POLLIN:
                    self.collect_task()
            else:
                if events.get(self.pull_socket) == zmq.POLLIN:
                    self.collect_task()
                if events.get(self.ctrl_socket) == zmq.POLLIN:
                    self.manage()

            gevent.sleep(0)

    def connect(self, push_addr = None,
                      pull_addr = None,
                      ctrl_addr = None):

        c = zmq.Context()

        # Pull tasks across manager
        pull_addr = zmq_addr(6666, transport='tcp')

        self.pull_socket = c.socket(zmq.PULL)
        self.pull_socket.bind(pull_addr)

        push_addr = zmq_addr(5555, transport='tcp')

        self.push_socket = c.socket(zmq.PUSH)
        self.push_socket.bind(push_addr)

        ctrl_addr = zmq_addr(7777, transport='tcp')

        self.ctrl_socket = c.socket(zmq.ROUTER)
        self.ctrl_socket.bind(ctrl_addr)

        store_addr = zmq_addr(8888, transport='tcp')

        self.store_socket = c.socket(zmq.REQ)
        self.store_socket.bind(store_addr)

    def start(self, timeout=None):
        gevent.signal(signal.SIGQUIT, gevent.shutdown)

        self.gen_bytecode()
        self.logging.info('Started Server')

        main = gevent.spawn(self.main_loop)
        heartbeat = gevent.spawn(self.heartbeat_loop)

        gevent.joinall([
            main,
            heartbeat,
        ])

        # Clean exit
        self.done()

    def done(self):
        for worker in self.workers:
            self.ctrl_socket.send_multipart([worker, 'done'])

    def manage(self):
        """
        Manage hearbeats on workers. Keep track of clients that
        are alive.
        """
        msg = self.ctrl_socket.recv_multipart()

    def _kill(self):
        gr = gevent.getcurrent()
        gr.kill()

    def results(self):
        if self.completed:
            return self.reduce_results
        else:
            return None

    def send_datum(self, command, key, data):
        self.push_socket.send(command, flags=zmq.SNDMORE)
        self.push_socket.send(str(key), flags=zmq.SNDMORE)

        if self.state == MAP:
            self.push_socket.send(data, copy=False)
        else:
            self.push_socket.send(srl.dumps(data))

    def send_command(self, command, payload=None):

        if payload:
            self.send_datum(command, *payload)
        else:
            self.push_socket.send(command)

    def start_new_task(self):
        action = self.next_task()
        if action:
            command, payload = action
            self.send_command(command, payload)

    def next_task(self):
        """
        The main work cycle, does all the distribution.
        """

        # -------------------------------------------
        if self.state == START:
            self.map_iter = self.datafn()

            if self.backend == MEMORY:
                self.map_results = defaultdict(list)
            elif self.backend == DKV:
                raise NotImplementedError
            else:
                import pdb; pdb.set_trace()
                raise NotImplementedError

            self.state = MAP
            self.logging.info('Mapping')

        # -------------------------------------------
        if self.state == MAP:
            try:
                map_key, map_item = next(self.map_iter)
                self.working_maps[str(map_key)] = map_item
                return 'map', (map_key, map_item)
            except StopIteration:
                self.logging.info('Shuffling')
                self.state = SHUFFLE

        # -------------------------------------------
        if self.state == SHUFFLE:
            self.reduce_iter = self.map_results.iteritems()
            self.working_reduces = set()
            self.reduce_results = {}

            if len(self.working_maps) == 0:
                self.logging.info('Reducing')
                self.state = PARTITION
            else:
                self.logging.debug('Remaining %s ' % len(self.working_maps))
                #self.logging.debug('Pending chunks %r' % self.working_maps.keys())

        # -------------------------------------------
        if self.state == PARTITION:
            # Normally we would define some sort way to balance the work
            # across workers ( key modulo n ) but ZMQ PUSH/PULL load
            # balances for us.
            self.state = REDUCE

        # -------------------------------------------
        if self.state == REDUCE:
            try:
                reduce_key, reduce_value = next(self.reduce_iter)
                self.working_reduces.add(reduce_key)
                return 'reduce', (reduce_key, reduce_value)
            except StopIteration:
                self.logging.info('Collecting')
                self.state = COLLECT

        # -------------------------------------------
        if self.state == COLLECT:
            if len(self.working_reduces) == 0:
                self.completed = True
                self.logging.info('Finished')
            else:
                self.logging.debug('Still collecting %s' % len(self.working_reduces))

    def collect_task(self):
        # Don't use the results if they've already been counted
        command = self.pull_socket.recv(flags=zmq.SNDMORE)

        if command == proto.CONNECT_WORKER:
            worker_id = self.pull_socket.recv()
            self.send_code(worker_id)

        elif command == proto.CONNECT_STORE:
            store_id = self.pull_socket.recv()
            store_ip = self.pull_socket.recv()
            print 'adding store', store_id, store_ip

            self.stores.add(store_id)
            self.ring.add_node(store_ip)

        # Maps Units
        # ----------

        elif command == proto.MAPCHUNK:
            key = self.pull_socket.recv()
            del self.working_maps[key]

        elif command == proto.MAPATOM:
            key = self.pull_socket.recv(flags=zmq.SNDMORE)
            tkey = self.pull_socket.recv(flags=zmq.SNDMORE)
            value = self.pull_socket.recv()

            self.map_results[tkey].extend(value)

        # Reduce Units
        # ------------

        elif command == proto.REDUCEATOM:
            key = self.pull_socket.recv(flags=zmq.SNDMORE)
            value = srl.loads(self.pull_socket.recv())

            # Don't use the results if they've already been counted
            if key not in self.working_reduces:
                return

            self.reduce_results[key] = value
            self.working_reduces.remove(key)

        else:
            assert 0, "Unknown chatter"

    def gen_bytecode(self):
        self.bytecode = (
            marshal.dumps(self.mapfn.func_code),
            marshal.dumps(self.reducefn.func_code),
        )

    def gen_llvm(self, mapfn, reducefn):
        raise NotImplementedError

    def send_code(self, worker_id):
        # A new worker
        if worker_id not in self.workers:
            self.logging.info('Worker Registered: %s' % worker_id)
            self.workers.add(worker_id)

            payload = ('bytecode', self.bytecode)
            # The worker_id uniquely identifies the worker in
            # the ZMQ topology.
            self.ctrl_socket.send_multipart([worker_id, srl.dumps(payload)])
            self.logging.info('Sending Bytecode to %s' % worker_id)
        else:
            self.logging.debug('Worker asking for code again?')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Verbose logging')
    parser.add_argument('config',  help='Configuration')
    parser.add_argument('--verbose', help='Verbose logging')

    args = parser.parse_args()

    config = configuration.load(args.config)

    # Python code
    if '.py' in args.path:
        path = args.path.replace('.py', '')
        fp, pathname, description = imp.find_module(path)
        mod = imp.load_module(args.path, fp, pathname, description)

    # BLIR code
    elif '.bl' in args.path:
        from blir import compile, Context
        path = args.path
        ast, env = compile(open(path).read())
        ctx = Context(env)
        mod = ctx.mod

    srv = Server(config, mod)
    srv.connect()
    srv.start()
