# NOTE: PYTHONHASHSEED=0 is useful for reproducible debugging

__all__ = (
    'PureStaticNode',
    'DataInjectorNode',
    'FlowBuilder',
    'ResourceTypeInfo',
)

from itertools import chain
from operator import itemgetter
from math import inf
from bisect import (
    bisect_left,
    bisect_right,
)
from numpy import (
    apply_along_axis,
    array,
    empty,
    zeros,
    ones,
    hstack,
    nonzero,
    pad,
)
from ..Base import exec_with_injection

class ResourceTypeInfo:
    __slots__ = ('type_cls', 'aux_info')
    def __init__(self, type_cls, **kwargs):
        self.type_cls = type_cls
        self.aux_info = dict(kwargs)
    def __repr__(self):
        desc = '<ResourceTypeInfo: type = {}'.format(self.type_cls)
        if self.aux_info:
            desc += ' aux = {}'.format(self.aux_info)
        return desc + '>'

class PureStaticNode:
    __slots__ = ('_contracts',)
    def __init__(self, *contracts):
        self._contracts = contracts
    def static_contracts(self):
        ''' For static contracts an index in the output list is the mode index '''
        return self._contracts
    def get_contract(self, mode_idx):
        ''' Returns (require-list, provide-list) for static/dynamic modes '''
        return self._contracts[mode_idx]
    def setup(self, *args, **kwargs):
        msg = '{} does not implement setup(), that is an error'.format(
            self.__class__.__qualname__,
        )
        raise NotImplementedError(msg)

class DataInjectorNode(PureStaticNode):
    __slots__ = ('_modes',)
    def __init__(self, *data_per_mode, **kwargs):
        types_provided = kwargs.get('types_provided', False)
        self._modes = []
        contracts = []
        for data_dict in data_per_mode:
            # selecting some order of given resource names
            key_order = list(data_dict.keys())
            # we need nothing and provide given resources in selected order
            contracts.append(([], key_order))
            # values in the dict could be: object OR (object, type_info)
            out_values = [data_dict[k] for k in key_order]
            if not types_provided:
                # with no type information given, query types of objects
                out_types = [ResourceTypeInfo(type(v)) for v in out_values]
            else:
                # decouple values and type information
                out_values, out_types = zip(*out_values)
            # for a single object we will drop tuple around it; thus, any output
            # like (single-object,) becomes just single-object
            if len(out_values) == 1:
                out_values = out_values[0]
            self._modes.append((lambda: out_values, out_types))
        PureStaticNode.__init__(self, *contracts)
    def setup(self, mode_id, input_types, output_mask):
        # here, we have no input types (see our contracts); output_mask is no
        # more than a recommendation and for the sake of simplicity we will
        # ignore it
        return self._modes[mode_id]

class _OrderGuard:
    def _reset_staging(self):
        self._staging_constraints = []
        self._staging = (
            empty((0,), dtype = 'int'),
            empty((0,), dtype = 'int'),
        )
    def export_constraints(self):
        return list(chain(*map(itemgetter(1), self._history)))
    def _append_to_staging(self, indices_array):
        # appending x-indices and y-indices arrays correspondingly
        self._staging = tuple(map(hstack, zip(self._staging, indices_array)))
    def __init__(self):
        self._map = empty((0, 0), dtype = 'bool')
        self._count = 0
        self._history = []
        self._reset_staging()
    def allocate_ids(self, count):
        if count <= 0:
            # the following code will do no harm for the zero-count request;
            # this case is here for the sake of performance and correctness for
            # less-than-zero case
            return []
        prev_count = self._count
        self._count += count
        # it is possible that we will reuse some idling space, we need to fill
        # it with "false" values before actually using it
        avail_space = self._map.shape[0]
        reinit_bound = min(self._count, avail_space)
        if prev_count < reinit_bound:
            reinit_range = slice(prev_count, reinit_bound)
            self._map[:, reinit_range] = False
            self._map[reinit_range, :] = False
        # checking for changes need to be done with the reachability table
        extra_space_needed = self._count - avail_space
        if extra_space_needed > 0:
            # the following represents padding sizes (before, after) per axis
            pds = [(0, extra_space_needed)] * 2
            self._map = pad(self._map, pds, 'constant', constant_values = False)
        return list(range(prev_count, self._count))
    def drop_id(self, node_id):
        if node_id + 1 != self._count:
            raise ValueError('drop_node is allowed only for the last node')
        self._count = node_id
    def stage_constraint(self, predecessor_id, successor_id):
        # check if the opposite edge is present in the map; if it is, then
        # we have a circular dependency which we can't tackle, so we need to
        # report the error
        if self._map[successor_id, predecessor_id]:
            return False
        # recording the constraint (to correctly export later)
        self._staging_constraints.append((predecessor_id, successor_id))
        # check if the constraint is already fulfilled (transitive);
        # in case it is, we keep record of the constraint, but no map update
        # is needed since reachability won't change
        if self._map[predecessor_id, successor_id]:
            return True
        # now we need to update reachability map according to the new edge; we
        # are adding an edge, so no reachability reduction is possible and new
        # routes will contain the new edge
        to_src = array(self._map[:, predecessor_id])
        from_dst = array(self._map[successor_id, :])
        to_src[predecessor_id] = True
        from_dst[successor_id] = True
        # if we can reach predecessor_id from some node A and some node B
        # from successor_id, then we can reach A from B (using the new edge);
        # here, we are selecting a submatrix which is defined by selecting
        # a number of rows and cols
        current_state = self._map[to_src, :][:, from_dst]
        # current_state contains all the routes which are possible now, but some
        # of them are already present; since we need to be able to revert
        # changes, we need to keep track of these routes
        actual_changes = nonzero(~current_state)
        # now the problem is that actual_changes contains indices which are
        # relative to the slice we have cut, not the full map; thus, we need
        # to map indices accordingly
        if actual_changes[0].shape[0] > 0:
            row_map = nonzero(to_src)[0]
            col_map = nonzero(from_dst)[0]
            actual_changes = (
                apply_along_axis(row_map.__getitem__, 0, actual_changes[0]),
                apply_along_axis(col_map.__getitem__, 0, actual_changes[1]),
            )
            self._append_to_staging(actual_changes)
            # now, we are changing reachability status where it is applicable;
            # note that current_state is actually a matrix slice, not a scalar
            self._map[actual_changes] = True
        return True
    def assert_clear_staging(self):
        if any(map(lambda v: v.shape[0] > 0, self._staging)):
            raise RuntimeError('staging is not clear when it is expected to be')
        if len(self._staging_constraints) > 0:
            raise RuntimeError('unexpected staged constraints')
    def reset(self):
        self._map[self._staging] = False
        self._reset_staging()
    def commit(self):
        self._history.append((self._staging, self._staging_constraints))
        self._reset_staging()
    def rollback(self):
        self.assert_clear_staging()
        # now we need to extract the last committed changes and revert them;
        # we can do it by pushing the last changes into staging and doing reset
        self._staging, self._staging_constraints = self._history.pop()
        self.reset()

class _UsageGuard:
    def __init__(self, order_guard):
        self._order_guard = order_guard
        self._applied_modes = {}
        self._steps = []
        self._prio_sorted_id = [] # prio-sorted step_id list
        self._prio_sorted_vals = [] # sorted prio values list
    def is_eligible(self, provider_and_mode):
        provider_idx, mode_idx = provider_and_mode
        applied_modes = self._applied_modes.get(provider_idx, set())
        return mode_idx not in applied_modes
    def push_step(self, provider_and_mode, prio):
        self._order_guard.assert_clear_staging()
        # allocating an id for the current step (provider & mode)
        step_id = self._order_guard.allocate_ids(1)[0]
        # trying to register order relations between the current and other steps
        sort_idx = bisect_left(self._prio_sorted_vals, prio)
        # for indices < sort_idx we have priorities < prio
        constraints_clash = False
        for other_id in self._prio_sorted_id[:sort_idx]:
            if not self._order_guard.stage_constraint(other_id, step_id):
                constraints_clash = True
                break
        # for indices >= sort_idx we have priorities >= prio, so we need to drop
        # equal-priority ones since we have no order relations for these
        if not constraints_clash:
            # we are continuing only if no constraints violations so far
            gt_prio_idx = bisect_right(self._prio_sorted_vals, prio, sort_idx)
            for other_id in self._prio_sorted_id[gt_prio_idx:]:
                if not self._order_guard.stage_constraint(step_id, other_id):
                    constraints_clash = True
                    break
        if constraints_clash:
            # we have detected a constraint violation, so we are dropping the
            # current step and returning failure to the caller
            self._order_guard.reset()
            self._order_guard.drop_id(step_id)
            return None
        # at this point, it is clear that no problems did arise because of the
        # given provider usage; now we can save the provider id and mode to
        # prevent using it again in the future
        provider_idx, mode_idx = provider_and_mode
        current_records = self._applied_modes.get(provider_idx, None)
        if current_records is None:
            self._applied_modes[provider_idx] = set([mode_idx])
        else:
            current_records.add(mode_idx)
        # we need to commit our changes we did to the order_guard
        self._order_guard.commit()
        # pushing our provider info into sorted arrays of the priorities
        # in order to use relevant order restrictions for new providers
        self._prio_sorted_vals.insert(sort_idx, prio)
        self._prio_sorted_id.insert(sort_idx, step_id)
        # and recording information which give us ability to rollback the step
        self._steps.append((provider_and_mode, sort_idx, step_id))
        return step_id
    def rollback(self):
        provider_and_mode, sort_idx, step_id = self._steps.pop()
        provider_idx, mode_idx = provider_and_mode
        self._applied_modes[provider_idx].remove(mode_idx)
        self._prio_sorted_vals.pop(sort_idx)
        self._prio_sorted_id.pop(sort_idx)
        self._order_guard.rollback()
        self._order_guard.drop_id(step_id)
    def export_ids(self):
        return dict(map(itemgetter(2, 0), self._steps))

# FIXME: handle the same configuration pulled by different order of targets?

class _TargetsTracker:
    def __init__(self, order_guard, unresolved, resolved = []):
        self._order_guard = order_guard
        self._pending = set(unresolved) # resources to be resolved
        self._done = set(resolved) # already resolved resources
        self._rollback = [] # list of (-done, -pending, +pending, drop_ids)
        self._id_resource = {} # mapping from resource name to versions ids
        self._id_identify = {} # mapping from id to resource name
        # target resources are already in the _pending set, we need to allocate
        # ids for them; that is because override_set code will assume that names
        # in _pending are pushed by some other provider and in this case they
        # have ids assigned
        self._allocate_ids(self._pending)
    def export_ids(self):
        return dict(self._id_identify)
    def resources(self):
        return list(self._id_resource.keys())
    def next_target(self):
        # there is no pop()-like method to access an element without removing it
        for e in self._pending:
            return e
        return None
    def is_completed(self):
        return len(self._pending) == 0
    def push_step(self, require_set, provide_set, override_set, step_id):
        self._order_guard.assert_clear_staging()
        # first, we check some restrictions which must be fulfilled;
        if len(provide_set & (self._done - self._pending)) > 0:
            # here, we are refusing to accept multiple providers for a resource,
            # because then it is unclear which one to use, whether we should
            # check that these version are equal or which calculation can be
            # omitted; though there is a case when a resource which we are
            # going to provide is allowed to be in the _done set: when resource
            # is overridden but an origin is missing; that means we know how to
            # do the output version (hence _done), but we need an input version
            # (hence _pending)
            return False
        # end-users of a resource are bound to the first version in the list, so
        # we need to ensure the first version is always the final version;
        # this is not true when a provider is processed earlier than overriders,
        # since then the first version in the list is bound to the provider and
        # any end-user will simply ignore the overridden version;
        # thus, we allow the following cases: 1) the resource is unseen so far,
        # then we do everything right, 2) the resource is overridden and awaits
        # input version, it is okay since overriders order is enforced by usage
        # tracker, 3) the resource is needed but yet not resolved; therefore,
        # the only bad case here is when the overridden resource is in _done
        # and not in _pending
        overridden_done = override_set & self._done
        overridden_pending = override_set & self._pending
        if not overridden_done.issubset(overridden_pending):
            return False
        staging_ids_order = []
        def _allocate(resources_names):
            if len(resources_names) > 0:
                nonlocal self, staging_ids_order
                staging_ids_order += self._allocate_ids(resources_names)
        del_done, del_pending, add_pending = [], [], []
        constraints_clash = False
        # processing provide list, registering unmet resources to track order
        _allocate(provide_set - self._id_resource.keys())
        for resource_name in provide_set:
            # when we are providing a resource, we bind to the last version in
            # the version list; since we are going backward from the required
            # resources to their providers and then further up, it means that
            # we are providing the earliest version of the resource; that is
            # because other versions are expected to be some overriders which
            # are working on top of the origin version (ours)
            downstream_id = self._id_resource[resource_name][-1]
            if not self._order_guard.stage_constraint(step_id, downstream_id):
                constraints_clash = True
                break
            # making corresponding changes with resolved/unresolved targets sets
            # and preparing rollback deltas to revert these changes when needed
            if resource_name not in self._done:
                # it is possible when we are providing the origin version for
                # some chain of overriders: then the resource is hanging in both
                # _pending and _done sets
                self._done.add(resource_name)
                del_done.append(resource_name)
            # when a resource is occurred to be in a provide_set, it is not
            # necessary needed (i.e. as a final target or a dependency)
            if resource_name in self._pending:
                self._pending.remove(resource_name)
                add_pending.append(resource_name)
        if not constraints_clash and len(require_set) > 0:
            # processing require list: note that is not possible for a resource
            # to be registered on the previous step since provide_set does not
            # intersect with require_set
            unseen_requirements = require_set - self._id_resource.keys()
            _allocate(unseen_requirements)
            for resource_name in require_set:
                # here we are binding on the first element in version list; that
                # is because we want to use the latest (for "forward" order)
                # version of the resource; it is different from overriders which
                # are going to override the earliest version
                upstream_id = self._id_resource[resource_name][0]
                if not self._order_guard.stage_constraint(upstream_id, step_id):
                    constraints_clash = True
                    break
            else:
                # no break encountered, so we are good to proceed;
                # adjusting resolved/unresolved sets and revert-lists
                for resource_name in unseen_requirements:
                    # other requirements are either resolved (done) or already
                    # requested by some other provider (pending)
                    self._pending.add(resource_name)
                    del_pending.append(resource_name)
        if not constraints_clash and len(override_set) > 0:
            # processing override list: this set has no intersections with the
            # previous two; here things are little bit different, since we will
            # deal with two versions of a single resource at the same time;
            # note that we are already rejected situations when the overridden
            # resource is in _done and not in _pending; then, !_done means that
            # there were no overriders so far, _pending means that there are
            # some users of the resource
            # we register new versions of all overridden resources
            _allocate(override_set)
            # but for !_pending we register additional ones (since then !_done
            # and the resource have not been seen yet, which means that we have
            # to allocate both input and output versions)
            _allocate(override_set - self._pending)
            for resource_name in override_set:
                # now we are extracting the last two versions: the latest one
                # is always the one we allocated (it is our input version to
                # be provided by someone else)
                output_id, input_id = self._id_resource[resource_name][-2:]
                if not self._order_guard.stage_constraint(input_id, step_id):
                    constraints_clash = True
                    break
                if not self._order_guard.stage_constraint(step_id, output_id):
                    constraints_clash = True
                    break
                # after the overrider is applied the resource name will appear
                # in both _done and _pending: we have provided a version for the
                # future use and requesting a version for our input
                if resource_name not in self._done:
                    self._done.add(resource_name)
                    del_done.append(resource_name)
                # if the resource is already in _pending then we must not remove
                # it from _pending on the way back (since it is also requested
                # by someone else)
                if resource_name not in self._pending:
                    self._pending.add(resource_name)
                    del_pending.append(resource_name)
        # if something went wrong along the way, we need to reverse everything
        # we have done prior the failure detection
        if constraints_clash:
            # reverting changes which were made to pending and done sets
            self._revert_tracking_step(del_done, del_pending, add_pending)
            # resetting staged constraints
            self._order_guard.reset()
            # removing new identifiers which were allocated
            self._deallocate_ids(staging_ids_order)
            return False
        # here everything is good, so we can merge the staged changes;
        # first, commit changes which were staged at the order_guard
        self._order_guard.commit()
        # then record information for being able to rollback the current step
        self._rollback.append(
            (del_done, del_pending, add_pending, staging_ids_order),
        )
        return True
    def rollback(self):
        last_step = self._rollback.pop()
        self._revert_tracking_step(*last_step[:3])
        self._order_guard.rollback()
        self._deallocate_ids(last_step[3])
    def _revert_tracking_step(self, del_done, del_pending, add_pending):
        for name in del_pending:
            self._pending.remove(name)
        for name in del_done:
            self._done.remove(name)
        for name in add_pending:
            self._pending.add(name)
    def _allocate_ids(self, resources_names):
        allocated_ids = self._order_guard.allocate_ids(len(resources_names))
        for resource_name, alloc_id in zip(resources_names, allocated_ids):
            self._id_identify[alloc_id] = resource_name
            present_ids = self._id_resource.get(resource_name, None)
            if present_ids is None:
                self._id_resource[resource_name] = [alloc_id]
            else:
                present_ids.append(alloc_id)
        return allocated_ids
    def _deallocate_ids(self, allocation_step):
        for resource_id in reversed(allocation_step):
            # notify ids allocator about the resource_id release
            self._order_guard.drop_id(resource_id)
            # preserve consistency in resource/id mappers
            resource_name = self._id_identify.pop(resource_id)
            resource_version_ids = self._id_resource[resource_name]
            removed_id = resource_version_ids.pop()
            if removed_id != resource_id:
                raise ValueError('reverting order of resources ids is violated')
            # since we are using keys() information to check whether at least
            # one version of the resource is available, we need to drop keys
            # which point to empty lists
            if len(resource_version_ids) == 0:
                del self._id_resource[resource_name]

class _ConstrainedIDWalker:
    def __init__(self, id_table_size, constraints, reserve_deptable = False):
        # it is possible that the dependency table might be needed somewhere
        # outside _ConstrainedIDWalker, so we offer an option to store and
        # export it
        self._table_size = id_table_size
        if reserve_deptable:
            self._deptable = self._make_deptable(id_table_size, constraints)
        else:
            self._constraints = constraints
    def __iter__(self):
        try:
            # if the deptable is reversed, then we will use the stored version;
            # otherwise, we will build one for our purposes only, in this case
            # the table will be released right after the last item generated
            deptable = self._deptable
        except AttributeError:
            deptable = self._make_deptable(self._table_size, self._constraints)
        notdone = ones(self._table_size, dtype = 'bool')
        # forming the first wave which indices do not have any dependencies
        first_wave = (~deptable.any(axis = 0)).nonzero()[0]
        class _IdWalkerIterator:
            def __init__(self):
                nonlocal deptable, notdone, first_wave
                self._deptable = deptable
                self._notdone = notdone
                self._wave = first_wave
            def __iter__(self):
                return self
            def __next__(self):
                if len(self._wave) == 0:
                    # releasing resources, this iterator is useless from now
                    del self._deptable, self._notdone
                    raise StopIteration
                next_wave = []
                drop_ids = []
                for curr_id in self._wave:
                    # recording that the current id is completed, then for all
                    # ids which have us as a dependency checking whether
                    # completion of the current id was the last unresolved
                    # prerequisite -- in this case we have a member of the
                    # next wave of ids
                    self._notdone[curr_id] = False
                    pending_users = self._deptable[curr_id, :]
                    if pending_users.any():
                        # gathering "depend on" indicators for pending_users,
                        # then matching them with non-done indicators, this way
                        # each "true" on output means there is at least one id
                        # which we are depending on and the id is not done yet
                        deps = self._deptable[:, pending_users]
                        has_unres_dep = (deps.T & self._notdone).any(axis = 1)
                        ids = (~has_unres_dep).nonzero()[0]
                        # these indices are relative to pending_users, fixing
                        revmapping = pending_users.nonzero()[0]
                        ids = apply_along_axis(revmapping.__getitem__, 0, ids)
                        next_wave.extend(ids)
                    else:
                        # this case corresponds to the case where our node is
                        # not actually used by anyone, so it is safe to delete
                        # it right away
                        drop_ids.append(curr_id)
                    # then for all our prerequisites we should check whether we
                    # were the last user; in this case we should drop the id
                    # of the dependency since it is not needed anymore
                    rq = self._deptable[:, curr_id]
                    # we get our prerequisites (rq), the following expression
                    # will indicate whether an id has at least one unresolved
                    # user; negation will give us "non-needed-anymore" ids
                    deps = (self._deptable[rq, :] & self._notdone).any(axis = 1)
                    ids = (~deps).nonzero()[0]
                    # these indices are relative to rq, need mapping back
                    revmapping = rq.nonzero()[0]
                    ids = apply_along_axis(revmapping.__getitem__, 0, ids)
                    drop_ids.extend(ids)
                # returning the old wave and preparing for the next step
                self._wave, next_wave = next_wave, self._wave
                return (next_wave, drop_ids)
        return _IdWalkerIterator()
    def _make_deptable(self, id_table_size, constraints):
        deptable = zeros((id_table_size, id_table_size), dtype = 'bool')
        for pred_idx, succ_idx in constraints:
            deptable[pred_idx, succ_idx] = True
        return deptable
    def export_deptable(self):
        return self._deptable

class _RegistersAllocator:
    def __init__(self):
        self._top_index = -1 # maximum allocated index so far
        self._free_rooms = set()
    def alloc_room(self):
        try:
            return self._free_rooms.pop()
        except KeyError:
            self._top_index += 1
            return self._top_index
    def free_room(self, idx):
        self._free_rooms.add(idx)
    def max_allocated_rooms(self):
        return self._top_index + 1

class _ChainStorage:
    __slots__ = ('_data',)
    def __init__(self, room_count):
        self._data = [None] * room_count
    def reset_data(self):
        # note that we can't drop the list instance itself since
        # we made closures on it inside wrapper functions;
        for data_idx in range(len(self._data)):
            self._data[data_idx] = None
    def get_data(self, indices = None):
        if indices is None:
            return list(self._data)
        return [self._data[i] for i in indices]
    def wrap_function(self, func, input_indices, output_indices):
        data_fmt_chk = lambda i: 'd[{}]'.format(i) if i is not None else '__'
        code = 'def ff(): {outs} = f({ins})'.format(
            outs = ','.join([data_fmt_chk(i) for i in output_indices]),
            ins = ','.join(['d[{}]'.format(i) for i in input_indices]),
        )
        return exec_with_injection(code, 'ff', [('f', func), ('d', self._data)])
    def wrap_deleter(self, index):
        d, i = self._data, index
        def deleter():
            d[i] = None
        return deleter

class _ChainTemplate:
    def __init__(self, rooms, targets):
        self._types = [None] * rooms
        self._rooms = rooms
        self._targets = targets
        self._gears = []
    def get_type_info(self, indices):
        return [self._types[i] for i in indices]
    def update_type_info(self, types_and_indices):
        for index, new_type_info in types_and_indices:
            if index is not None:
                self._types[index] = new_type_info
    def drop_type_info(self):
        self._types = None
    def add_chainer_call(self, chainer_member, *args, **kwargs):
        self._gears.append((chainer_member, args, kwargs))
    def assemble(self):
        chainer = _ChainStorage(self._rooms)
        invokation_list = []
        for chainer_func, args, kwargs in self._gears:
            invokation_list.append(chainer_func(chainer, *args, **kwargs))
        def _execute_chain():
            for do_step in invokation_list:
                do_step()
            data = chainer.get_data(self._targets)
            chainer.reset_data()
            return data
        return _execute_chain

class FlowBuilder:
    def __init__(self):
        self._nodeobjs = [] # providers objects storage
        self._nodeprio = [] # provider index -> provider priority
        self._next_prio = 0 # the next priority to use by default
    def register(self, provider_obj, priority = None):
        # checking for required functions to be present
        contract_modes_supported = map(
            lambda name: hasattr(provider_obj, name),
            ['static_contracts', 'dynamic_contracts'],
        )
        if not any(contract_modes_supported):
            msg = 'node {} has no contract listing interface'
            raise RuntimeError(msg.format(provider_obj))
        # storing the provider itself and setting its priority
        self._nodeobjs.append(provider_obj)
        if priority is None:
            # default priority is matching the registration order
            self._nodeprio.append(self._next_prio)
            self._next_prio += 1
        else:
            # keeping the order correspondence for default priorities
            self._nodeprio.append(priority)
            self._next_prio = max(priority + 1, self._next_prio)
        # NOTE: race condition
        return len(self._nodeobjs) - 1
    def _find_build_steps(self, targets, lookup_providers):
        viable_sequences = [] # viable sequences of providers to calc targets
        step_options = [] # provider, mode options left unexplored per step
        order_guard = _OrderGuard()
        track = _TargetsTracker(order_guard, targets)
        usage = _UsageGuard(order_guard)
        rolling_back_now = False
        while True:
            # rolling_back_now here means that the previous iteration ended up
            # with no further processing possible or with a complete sequence
            # of actions which produced the required targets; thus, we are going
            # back in time to check out other provider options
            if rolling_back_now:
                # it is possible that the options list contains no elements
                # in this case we will end up with rolling back further
                options = step_options.pop()
                # single-step rollback is completed; it might be triggered again
                # if no viable options will be found on this step
                rolling_back_now = False
            else:
                # here we are moving forward, i.e. picking some resource that
                # needs to be resolved and looking up for providers which can
                # give us the resource
                target = track.next_target()
                options = lookup_providers(target, track.resources())
                # drop the options we can't use since they are already used,
                # that is to avoid looping for override-providers
                options = list(filter(usage.is_eligible, options))
            # now we will try to use an option which satisfies all our
            # requirements, i.e. the target providers restrictions (like not
            # to use multiple providers which will give us the same resource
            # independently) and the order restrictions (these ones are
            # essentially of two kinds: providers priorities and "user is
            # after provider")
            for idx, option in enumerate(options):
                prov_idx, mode_idx = option
                # to work with order control we need to register a provider id
                step_id = usage.push_step(option, self._nodeprio[prov_idx])
                if step_id is None:
                    # provider priorities were violated by this move,
                    # continuing search using other options
                    continue
                contract = self._nodeobjs[prov_idx].get_contract(mode_idx)
                require, provide = map(set, contract)
                override = require & provide
                require -= override
                provide -= override
                if not track.push_step(require, provide, override, step_id):
                    # resource ordering is violated; we continue our search,
                    # but the current step id needs to be cancelled
                    usage.rollback()
                    continue
                # at this point we confirmed that the selected provider matches
                # all requirements; we can check if everything is done now to
                # avoid unnecessary iteration that follows
                if track.is_completed():
                    # FIXME ensure "usage" agrees
                    # everything is done, recording calculated providers
                    # sequence to the output storage
                    viable_option = (
                        usage.export_ids(), # providers mapping
                        track.export_ids(), # resources mapping
                        order_guard.export_constraints(),
                    )
                    viable_sequences.append(viable_option)
                    # since we want to try out another options we have not
                    # explore yet, we need to rewind our work; to do so we can
                    # pretend the current option is no good for us (rolling back
                    # the last step) then we will go on with our search
                    track.rollback()
                    usage.rollback()
                    continue
                # here, we have found a viable step to move forward, but it is
                # not the last step; since we are doing DFS-like search, we just
                # save other options (to explore later) and move forward with
                # the current choice
                step_options.append(options[idx + 1 :])
                break
            else:
                # no viable provider was found (or we encountered completed
                # configurations), we are in a dead-end and should rollback
                # to the previous step in order to try other options; note that
                # we also jump here if the upper loop is done over empty list
                if len(step_options) == 0:
                    # in this case we have tried everything we can
                    break
                track.rollback()
                usage.rollback()
                rolling_back_now = True
        return viable_sequences
    def _build_actions(self, providers, resources, constraints, targets_list):
        # assuming that ids are allocated continuously from 0 to max-id
        alloc_id_range = max(max(providers.keys()), max(resources.keys())) + 1
        walker = _ConstrainedIDWalker(alloc_id_range, constraints, True)
        deptable = walker.export_deptable()
        # using the walker we will get chunks of ids, which are grouped in the
        # way such that the first group does not depend on anything and every
        # other group depends only on ids in previous groups; note that id can
        # be related to a resource and or a provider, applied logic differs
        def _lookup_deps(node_id):
            return deptable[:, node_id].nonzero()[0]
        def _lookup_users(node_id):
            return deptable[node_id, :].nonzero()[0]
        def _split_group(id_group):
            resources_group, providers_group = [], []
            for curr_id in id_group:
                try:
                    resources_group.append((curr_id, resources[curr_id]))
                except KeyError:
                    providers_group.append((curr_id, providers[curr_id]))
            return resources_group, providers_group
        def _extract_resources(id_group):
            resources_mapping = {}
            for curr_id in id_group:
                try:
                    # NOTE: here we are not going to handle a situation where
                    # multiple requirement ids are mapping to the same resource
                    resources_mapping[resources[curr_id]] = curr_id
                except KeyError:
                    continue
            return resources_mapping
        targets = set(targets_list) # quick lookup
        reg = _RegistersAllocator()
        reg_alloc_ids = {} # object id -> allocated registry index
        action_list, delayed_workers, delayed_drops = [], [], []
        for id_group_add, id_group_del in chain(walker, [([], [])]):
            add_resources, add_providers = _split_group(id_group_add)
            del_resources = _split_group(id_group_del)[0]
            # processing workers of the current step; since their output nodes
            # are placed in the next step, we will resolve only input part of
            # the contract and the rest will be processed on the next iteration
            new_delayed_workers = []
            for curr_prov_id, prov_info in add_providers:
                # looking up the contract of the provider
                prov_id, mode_id = prov_info
                contract = self._nodeobjs[prov_id].get_contract(mode_id)
                # matching require-part of the contract with dependencies ids
                deps_ids = _extract_resources(_lookup_deps(curr_prov_id))
                input_ids = map(deps_ids.__getitem__, contract[0])
                # these are ids of the objects we need on our input; now we
                # need to map them into registry indices
                input_regs = tuple(map(reg_alloc_ids.__getitem__, input_ids))
                # output registers are unknown at this point, since they will
                # be allocated only on the next step (since output resources
                # ids are dependent on the provider id)
                res_out_ids = _extract_resources(_lookup_users(curr_prov_id))
                output_ids = tuple(map(res_out_ids.__getitem__, contract[1]))
                # now we are ready to push the provider info into pending list
                new_delayed_workers.append((prov_info, input_regs, output_ids))
            # now we are going to process addition and removal of resources;
            # it is possible that a resource is marked for addition and removal
            # in the same step: a resource is provided but not used by anyone;
            # we are not going to allocate and process these resources with an
            # exception for target resources which are usually not used by
            # anyone but still required to store
            del_resources = set(e for e, v in del_resources if v not in targets)
            output_placeholder_drops = []
            for resource_id, resource_name in add_resources:
                if resource_id in del_resources:
                    # here we are placing mapping not to an registry index but
                    # to None; assigner function will ignore any provider output
                    # which is configured to store element in None-index
                    output_placeholder_drops.append(resource_id)
                    del_resources.remove(resource_id)
                    alloc_idx = None
                else:
                    alloc_idx = reg.alloc_room()
                reg_alloc_ids[resource_id] = alloc_idx
            new_delayed_drops = del_resources
            # we have new resources allocated, so we can proceed with processing
            # of delayed providers (they were waiting for output ids allocation)
            for prov_info, input_regs, output_ids in delayed_workers:
                output_regs = tuple(map(reg_alloc_ids.__getitem__, output_ids))
                action_list.append((prov_info, input_regs, output_regs))
            # after providers are processed we don't need "none" placeholders
            # in registry allocation mapping (these resources by definition have
            # no actual users, so it is safe to remove mapping right away)
            for resource_id in output_placeholder_drops:
                del reg_alloc_ids[resource_id]
            # at this point in the action list the delayed workers are done, so
            # we can release the resources which were required as workers inputs
            for resource_id in delayed_drops:
                used_index = reg_alloc_ids.pop(resource_id)
                reg.free_room(used_index)
                # this action will push None into the specified index causing
                # the decrement of usage counter and hopefully memory release
                action_list.append((None, None, used_index))
            # now we are passing the current delayed lists on the next iteration
            delayed_workers = new_delayed_workers
            delayed_drops = new_delayed_drops
        # the last action is to gather resources from registry space (array), so
        # we're providing indices in the order which corresponds to targets_list
        resource_index_mapping = {}
        for curr_id, alloc_idx in reg_alloc_ids.items():
            # NOTE: here we are not going to handle a situation where there are
            # present multiple ids/indices which correspond to a single resource
            resource_index_mapping[resources[curr_id]] = alloc_idx
        indices = tuple(map(resource_index_mapping.__getitem__, targets_list))
        return (action_list, reg.max_allocated_rooms(), indices)
    def _build_contracts_lookup(self):
        # construct resource providers index for static contracts
        # and a list of nodes with dynamic contracts capabilities
        static_providers_index = {}
        dynamic_providers = []
        for node_idx, node_obj in enumerate(self._nodeobjs):
            if hasattr(node_obj, 'dynamic_contracts'):
                dynamic_providers.append(node_idx)
            try:
                static_contracts = node_obj.static_contracts()
            except AttributeError:
                continue
            for mode_idx, contract in enumerate(static_contracts):
                our_option = (node_idx, mode_idx)
                for resource in contract[1]: # "provide" part of contract
                    providers = static_providers_index.get(resource, None)
                    if providers is None:
                        static_providers_index[resource] = [our_option]
                    else:
                        providers.append(our_option)
        def _lookup_providers(resource_name, available_res):
            options = static_providers_index.get(resource_name, [])
            for prov_idx in dynamic_providers:
                node = self._nodeobjs[prov_idx]
                # dynamic_contracts() will return mode_id's, need to prepend
                # index of provider to form sane option record
                modes = node.dynamic_contracts(resource_name, available_res)
                options += map(lambda mode_id: (prov_idx, mode_id), modes)
            return options
        return _lookup_providers
    def construct(self, targets_list):
        configuration_sets = self._find_build_steps(
            targets_list,
            self._build_contracts_lookup(),
        )
        results_list = []
        for config in configuration_sets:
            actions, rooms, targets = self._build_actions(*config, targets_list)
            chainer_template = _ChainTemplate(rooms, targets)
            for prov_inf, in_ids, out_ids in actions:
                if prov_inf is None:
                    # note that out_ids in this case is actually just an index
                    chainer_template.add_chainer_call(
                        _ChainStorage.wrap_deleter,
                        out_ids,
                    )
                else:
                    prov_idx, prov_mode = prov_inf
                    nodeobj = self._nodeobjs[prov_idx]
                    outmask = [x is not None for x in out_ids]
                    try:
                        worker, types = nodeobj.setup(
                            prov_mode,
                            chainer_template.get_type_info(in_ids),
                            outmask,
                        )
                    except RuntimeError: # FIXME: change RuntimeError
                        # type conflicts in setup
                        chainer_template = None
                        break
                    chainer_template.update_type_info(zip(out_ids, types))
                    chainer_template.add_chainer_call(
                        _ChainStorage.wrap_function,
                        worker,
                        in_ids,
                        out_ids,
                    )
            if chainer_template is not None:
                out_type_info = tuple(chainer_template.get_type_info(targets))
                chainer_template.drop_type_info()
                results_list.append((chainer_template, out_type_info))
        return results_list
