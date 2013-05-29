import os
import sys
import atexit
import fcntl
import signal

def daemonize(pidfile, stdin='/dev/null',
                       stdout='/dev/null',
                       stderr='/dev/null'):

    if os.path.exists(pidfile):
        raise RuntimeError('Already running')

    try:
        if os.fork() > 0:
            raise SystemExit(0)
    except OSError:
        raise RuntimeError('fork #1 failed.')

    os.chdir('/')
    os.umask(0)
    os.setsid()

    # doublefork
    try:
        if os.fork() > 0:
            raise SystemExit(0)
    except OSError:
        raise RuntimeError('fork #2 failed.')

    sys.stdout.flush()
    sys.stderr.flush()

    with open(stdin, 'rb', 0) as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(stdout, 'ab', 0) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with open(stderr, 'ab', 0) as f:
        os.dup2(f.fileno(), sys.stderr.fileno())

    with open(pidfile,'w') as f:
        f.write(str(os.getpid()))
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    atexit.register(lambda: os.remove(pidfile))

    def sigterm_handler(signo, frame):
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)
