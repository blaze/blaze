import sys, os.path, SimpleXMLRPCServer


def obedient_executor(module_name, module_source, args=()):
    """
    >>> obedient_executor("blah", "def run(): return 4*5")
    20
    >>> obedient_executor("blah", "")
    20
    >>> obedient_executor("blah2", "def run(x, y): return x*y", (7,8))
    56
    """
    if module_source != "":
        with open(os.path.join(".", str(module_name) + ".py"), 'w') as fp:
            fp.write(module_source)
    supplied_module = __import__(module_name)
    result = supplied_module.run(*args)
    return result


def ping():
    return "pong"



if __name__ == '__main__':
    listen_port = 9002
    if len(sys.argv) > 1:
        listen_port = int(sys.argv[1])

    server = SimpleXMLRPCServer.SimpleXMLRPCServer(("0.0.0.0", listen_port))
    server.register_function(obedient_executor)
    server.register_function(ping)

    print "Listening on port {}".format(listen_port)
    server.serve_forever()

