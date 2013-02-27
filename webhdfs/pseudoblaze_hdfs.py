import xmlrpclib, multiprocessing


def _doacross_action(module_name, module_source, node_uri, file_uris, \
        results, fixed_args=()):
    """Intended to be invoked inside a single thread with 'results' as
    multiprocessing.Queue() instance."""
    proxy = xmlrpclib.ServerProxy(node_uri)
    for file_uri in file_uris:
        out = proxy.obedient_executor(module_name, module_source, \
                (file_uri,) + fixed_args)
        results.put((node_uri, file_uri, out))



class HDFSClusterSupervisor:
    """Prototype supervisor for performing HDFS-data-local code execution."""

    def __init__(self, desired_worker_nodes=[], verify_nodes_now=True):
        self.worker_nodes = desired_worker_nodes
        if verify_nodes_now:
            self.verify_active_nodes()


    def add_worker_node(self, node_uri, verify_node_now=True):
        was_added = False
        if verify_node_now:
            if self._ping_node(node_uri):
                self.worker_nodes += [node_uri]
                was_added = True
        else:
            self.worker_nodes += [node_uri]
            was_added = True
        return was_added


    def doacross(self, module_name, module_source, nodes_files_uris, \
            fixed_args=()):
        """Invokes the supplied module (with given name and source) run()
        method on each node specified in the nodes_files_uris list of tuples,
        where items in the nodes_files_uris list are of the form
        (node_uri, [file_uri0, file_uri1, ...]).  All invocations of run() on
        a particular node are performed from the same thread and each
        distinct node gets its own thread for launching runs there."""
        manager = multiprocessing.Manager()
        results = manager.Queue()
        pool = multiprocessing.Pool(len(nodes_files_uris))
        for (node_uri, file_uris) in nodes_files_uris:
            pool.apply_async(_doacross_action, \
                    (module_name, module_source, node_uri, file_uris, \
                    results, fixed_args))
        pool.close()
        pool.join()

        result_items = []
        while results.empty() == False:
            result_items.append(results.get(block=False))
        return result_items


    def build_map_files_to_local_uris(self, filepaths):
        """Build a map of uris to HDFS Datanodes where a given file is
        actually stored (redundantly, in some cases) by querying WebHDFS
        on each worker node."""
        mapped_files = {}
        if type(filepaths) == type(""):
            # Interpret string as directory name to walk through files within.
            # TODO: Permit a directory name and discover files within
            filepaths = []  # TODO: build this by walking
            raise ValueError("Unimplemented: Walking through a directory.")

        # The hdfs_filemapper module abuses the 'file_uris' by passing
        # HDFS file paths in their stead; 'out' (as seen inside #doacross())
        # contains the optimal Datanode uri for reading that file,
        # according to the worker node (WebHDFS there) that was queried.
        mapped_optimal_uris = self.doacross("hdfs_filemapper", "", \
                zip(self.worker_nodes, len(self.worker_nodes)*[filepaths]))
        for (node_uri, filename, file_uri) in mapped_optimal_uris:
            if filename not in mapped_files:
                mapped_files[filename] = []
            if file_uri != 'None':
                mapped_files[filename].append((node_uri, file_uri))

        return mapped_files


    def verify_active_nodes(self):
        """Walk through and keep nodes in the 'worker_nodes' list that
        respond to a ping in a timely manner."""
        self.worker_nodes = [ node_uri for node_uri in self.worker_nodes \
                              if self._ping_node(node_uri) ]
        return len(self.worker_nodes)


    def _ping_node(self, node_uri):
        """Returns True if specified node responds to a ping, else False."""
        got_response = True
        try:
            proxy = xmlrpclib.ServerProxy(node_uri)
            proxy.ping()
        except:
            got_response = False
        return got_response



class HDFSClusterJob:
    def __init__(self, sup, filepaths=[], module_name=None, module_source=None):
        self._cluster_supervisor = sup
        self._module_name, self._module_source = module_name, module_source
        self._previously_run_with_same_module = False
        self.filepaths = filepaths
        self.execution_plan = []
        if len(filepaths) != 0:
            self.prepare_execution_plan()


    def prepare_execution_plan(self):
        mapped_files = self._cluster_supervisor.build_map_files_to_local_uris(self.filepaths)
        node_tasks = {}
        # Employs sub-optimal single pass with no global optimization.
        for filename in mapped_files:
            for (node_uri, file_uri) in mapped_files[filename]:
                # Initialize node_tasks.
                if node_uri not in node_tasks:
                    node_tasks[node_uri] = []
            chosen_node_uri, chosen_file_uri = sorted([ \
                    (len(node_tasks[node_uri]), (node_uri, file_uri)) \
                    for (node_uri, file_uri) in mapped_files[filename] ])[0][1]
            node_tasks[chosen_node_uri].append(chosen_file_uri)
        self.execution_plan = node_tasks.items()


    def set_module_from_string(self, module_name, module_source):
        self._module_name, self._module_source = module_name, module_source
        self._previously_run_with_same_module = False


    def set_module_from_file(self, filename):
        with open(filename, 'r') as fp:
            self._module_source = fp.read()
        self._module_name = filename.split('.')[0]
        self._previously_run_with_same_module = False


    def run(self, fixed_args=()):
        module_source = self._module_source
        if self._previously_run_with_same_module:
            # Reuse the cached copy on the nodes, skip resending over wire.
            module_source = ""
        return self._cluster_supervisor.doacross(self._module_name, \
                module_source, self.execution_plan, fixed_args)


