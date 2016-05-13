__all__ = (
    'DependencyFlowManager',
)
from copy import deepcopy
from itertools import count

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
        #   required-list is a list of [resource-name, resource-version-idx]
        #   provided-list is a list of resource-name's.
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
            msg = '"{}" node does not comply with a node interface'
            raise RuntimeError(msg.format(node))
        # registering resources provided by the node
        for mode_id, requires, provides in declared_contracts:
            curr_provider_id = (expected_node_idx, mode_id)
            # we need a list to ensure a fixed order
            overridden = list(set(requires).intersection(set(provides)))
            # require-list by default uses the last available version (-1 idx)
            requires_index_lookup = dict(zip(requires, count()))
            requires = list(map(lambda x: [x, -1], requires))
            # just to be clearer about resources override -- imagine an example
            # of a dataset filtering; thus, the node gets the current version of
            # the dataset and produces an updated dataset which will get the
            # same resource name; this way, the dataset transparently includes
            # the given filtering for the further use
            if len(overridden) > 0:
                # here, we must have previous version to work with
                version_indices = []
                for resource_name in overridden:
                    version_count = len(self._resources.get(resource_name, []))
                    if version_count == 0:
                        msg = '{} needs {}, but nobody has declared it yet'
                        raise RuntimeError(msg.format(node, resource_name))
                    version_indices.append(version_count - 1)
                for resource_name, ver_idx in zip(overridden, version_indices):
                    resource_versions = self._resources.get(resource_name)
                    # we are adding a new version that uses the previous ones
                    resource_versions.append([curr_provider_id])
                    # we need to override the required version in "requires"
                    # list in order to avoid looping to ourselves
                    override_idx = requires_index_lookup[resource_name]
                    requires[override_idx][1] = ver_idx
            # processing provided resources which are not overridden
            for resource_name in (set(provides) - set(overridden)):
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
    def construct_graph(self, target_resources_list):
        pass
