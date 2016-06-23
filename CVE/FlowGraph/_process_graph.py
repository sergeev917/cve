__all__ = (
    'DependencyFlowManager',
)
from copy import deepcopy
from itertools import count, chain
from operator import itemgetter

class ChainedProviders:
    # NOTE: preserve_resources are names with versions
    def __init__(self, resource_usage_tracker, preserve_resources = []):
        self._storage = []
        self._worker_list = []
        self._resource_mapper = {}
        self._resource_tracker = dict(resource_usage_tracker)
        self._storage_size = 0
        self._preserve_resources = set(preserve_resources)
    def push_worker(self, prov_id, worker, input_resources, output_resources):
        self._resource_mapper.update(
            zip(output_resources, count(self._storage_size))
        )
        self._storage_size += len(output_resources)
        resource_idx_getter = self._resource_mapper.__getitem__
        in_indices = map(resource_idx_getter, input_resources)
        out_indices = map(resource_idx_getter, output_resources)
        def invoke_worker():
            worker_output = worker(*map(self._storage.__getitem__, in_indices))
            print('invoke_worker: output is {}'.format(worker_output))
            for out_idx, out_value in zip(out_indices, worker_output):
                self._storage[out_idx] = out_value
        self._worker_list.append(invoke_worker)
        # if we are the last users of our input resources, we should remove them
        for resource in input_resources:
            users = self._resource_tracker[resource]
            users.remove(prov_id)
            if len(users) == 0 and resource not in self._preserve_resources:
                def remove_resource():
                    print('removing {}'.format(resource))
                    self._storage[resource_idx_getter(resource)] = None
                self._worker_list.append(remove_resource)
    def __call__(self):
        print('resource mapper: {}'.format(self._resource_mapper))
        self._storage = [None] * self._storage_size
        for worker in self._worker_list:
            worker()
    def get_resource(self, resource, version = -1):
        return self._storage[self._resource_mapper[(resource, version)]]
    def get_available_resources(self):
        resource_values = map(
            lambda name, idx: (name, self._storage[idx]),
            self._resource_mapper,
        )
        vals = filter(lambda name, value: value is not None, resource_values)
        return dict(vals)

class DependencyFlowManager:
    def __init__(self):
        # resources is a dictionary:
        #   resource name (an object to be provided) -> list of versions;
        #   each version is a list of possible providers:
        #       (provider-node-idx, provider-mode-idx)
        self._resources = {}
        # nodes is a list of resource providers (see provider-node-idx above):
        #   each resource provider is a tuple:
        #     (provider-object, modes-desc):
        #       modes-desc = dict: mode-id -> (required-list, provided-list)
        #   required-list and provided-list are lists of
        #       (resource-name, resource-version-idx)
        self._nodes = []
    def add_node(self, node):
        # reserving space for the current node, will replace the last obj
        expected_node_idx = len(self._nodes)
        contracts_to_register = {} # mode-id -> require/provide
        # extracting contracts from the node: each contract (recipe) here
        # is a tuple: (mode-id, require-list, provide-list)
        # note that mode-id must be valid for dynamic contracts too. A way
        # to achieve this: use the same interface for all possible arguments
        # of dynamic contract or store some instructions for a new mode id
        declared_contracts = []
        interface_compliance = False
        try:
            declared_contracts = node.dynamic_recipes(self._resources.keys())
            interface_compliance = True
        except AttributeError:
            pass
        try:
            declared_contracts += node.static_recipes()
            interface_compliance = True
        except AttributeError:
            pass
        if not interface_compliance:
            msg = '"{}" node does not implement a node interface'
            raise RuntimeError(msg.format(node))
        # registering resources provided by the node
        for mode_id, requires, provides in declared_contracts:
            curr_provider_id = (expected_node_idx, mode_id)
            # we need a list to ensure a fixed order
            provision_set = set(provides)
            overridden = set(requires).intersection(provision_set)
            unseen_provisions = provision_set - overridden
            overridden = list(overridden)
            del provision_set
            # require-list by default uses the last available version (-1 idx)
            requires_index_lookup = dict(zip(requires, count()))
            # by default, we are requiring and providing the last version
            requires = list(map(lambda x: (x, -1), requires))
            provides = list(map(lambda x: (x, -1), provides))
            # just to be clearer about resources override -- imagine an example
            # of a dataset filtering; thus, the node gets the current version of
            # the dataset and produces an updated dataset which will get the
            # same resource name; this way, the dataset transparently includes
            # the given filtering for the further use
            if len(overridden) > 0:
                # in the first pass we are getting the last version indices
                version_indices = [] # last versions of overridden resources
                for resource_name in overridden:
                    prev_versions = self._resources.get(resource_name, [])
                    version_count = len(prev_versions)
                    if version_count == 0:
                        msg = '{} needs {}, but nobody has declared it yet'
                        raise RuntimeError(msg.format(node, resource_name))
                    current_version = version_count - 1
                    version_indices.append(current_version)
                    # also, we are overriding the currently used version (which
                    # is set to -1) to an actual current version
                    for prov_idx, mode_id in prev_versions[-1]:
                        prov_contracts = self._nodes[prov_idx][1]
                        prov_requires, prov_provides = prov_contracts[mode_id]
                        # searching the overridden resource in provision list
                        targets = filter(
                            lambda d: d[1][0] == resource_name,
                            enumerate(prov_provides),
                        )
                        # extracting indices
                        targets = list(map(itemgetter(0), targets))
                        assert len(targets) == 1
                        # setting the actual version
                        prov_provides[targets[0]] = (
                            resource_name,
                            current_version,
                        )
                # in the second pass we are setting our requirements to the
                # last versions of required resources from the first pass
                for resource_name, ver_idx in zip(overridden, version_indices):
                    resource_versions = self._resources[resource_name]
                    # we are adding a new version that uses the previous ones
                    resource_versions.append([curr_provider_id])
                    # we need to override the required version in "requires"
                    # list in order to avoid an input dependency on your output
                    override_idx = requires_index_lookup[resource_name]
                    requires[override_idx] = (resource_name, ver_idx)
            # processing provided resources which are not overridden
            for resource_name in unseen_provisions:
                if resource_name not in self._resources:
                    # one new version with the single way to get it
                    self._resources[resource_name] = [[curr_provider_id]]
                else:
                    # or push an alternative to the last version
                    self._resources[resource_name][-1].append(curr_provider_id)
            contracts_to_register[mode_id] = (requires, provides)
        # updating providers (with versions specified for overridden names)
        self._nodes.append((node, contracts_to_register))
        assert len(self._nodes) == expected_node_idx + 1
        return expected_node_idx
    def construct_graph(self, target_resources):
        found_configs = []
        # for target resource names the last version is selected
        target_resources = list(map(lambda i: (i, -1), target_resources))
        available_resources = set()
        providers_deployed = []
        _target_repr = lambda target: 'target resource {}@[{}]'.format(*target)
        # passing over graph to produce configurations list
        def _find_providers():
            nonlocal available_resources, target_resources
            if len(target_resources) == 0:
                found_configs.append(list(providers_deployed))
                return
            current_resource = target_resources.pop()
            print('resolving {}'.format(_target_repr(current_resource)))
            res_name, res_version = current_resource
            providers_list = self._resources[res_name][res_version]
            if len(providers_list) == 0:
                print('-- no available providers for {}'.format(res_name))
                return
            for provider_idx, provider_mode in providers_list:
                print('using provider: {}, {}'.format(provider_idx, provider_mode))
                require, provide = self._nodes[provider_idx][1][provider_mode]
                additional_targets = set(require) - available_resources
                print('  add targets: {}'.format(additional_targets))
                new_resources = set(provide) - available_resources
                if len(new_resources) != len(provide):
                    print('  -- providers overlap is found')
                    return
                providers_deployed.append((provider_idx, provider_mode))
                print('  new resources: {}'.format(new_resources))
                available_resources.update(new_resources)
                target_resources_rollback = len(target_resources)
                target_resources += additional_targets
                _find_providers()
                print('  rolling back targets: {}'.format(additional_targets))
                target_resources = target_resources[:target_resources_rollback]
                print('  rolling back resources: {}'.format(new_resources))
                available_resources.difference_update(new_resources)
                providers_deployed.pop()
            target_resources.append(current_resource)
        _find_providers()
        print(found_configs)
        if len(found_configs) == 0:
            raise RuntimeError('unable to construct the requested data flow')
        # use configurations to produce worker function
        def _build_levels(providers):
            levels = []
            unused_providers = list(providers)
            available_resources = set()
            resource_users = dict()
            while len(unused_providers) > 0:
                current_level = []
                new_resources = set()
                for idx, prov in reversed(list(enumerate(unused_providers))):
                    prov_idx, mode_id = prov
                    require, provide = self._nodes[prov_idx][1][mode_id]
                    if set(require).issubset(available_resources):
                        current_level.append(prov)
                        unused_providers.pop(idx)
                        new_resources.update(provide)
                        for required_item in require:
                            resource_users[required_item].add(prov)
                        for provided_item in provide:
                            resource_users[provided_item] = set()
                if len(current_level) == 0:
                    raise RuntimeError('providers graph has looped')
                levels.append(current_level)
                available_resources.update(new_resources)
            del available_resources, unused_providers
            # now moving from 'need nothing' providers to the target ones;
            # negotiating the input/output types on the way.
            do_all_builder = ChainedProviders(resource_users)
            configured_types = {}
            for providers_level in levels:
                for provider in providers_level:
                    prov_idx, mode_id = provider
                    obj, contracts = self._nodes[prov_idx]
                    require, provide = contracts[mode_id]
                    in_types = map(configured_types.__getitem__, require)
                    try:
                        worker, out_types = obj.configure_for(
                            mode_id,
                            in_types,
                            parent = self,
                        )
                    except RuntimeError: # FIXME
                        raise RuntimeError from None
                    for name, config_type in zip(provide, out_types):
                        configured_types[name] = config_type
                    do_all_builder.push_worker(
                        provider,
                        worker,
                        require,
                        provide,
                    )
            return do_all_builder
        do_all = _build_levels(found_configs[0])
        do_all()
        print(do_all._storage)




