__all__ = (
    'BoundingBoxSampleAnnotation',
    'BoundingBoxLoader',
)

from operator import itemgetter
from numpy import (
    int32,
    ndarray,
    array,
    hstack,
    column_stack,
)
from ...Base import (
    NonApplicableLoader,
    LoaderMissingData,
)

class BoundingBoxSampleAnnotation:
    '''The class stores single matrix which represents all bounding boxes on
       given sample (which is supposed to be a single image). The data layout
       is designed to accelerate matrix operations such as selecting max or
       min on one feature channel (for instance, x_min or y_max), not one
       bounding box. So note that the data is stored row-wise, a bounding box
       is a column in the matrix.'''
    __slots__ = ('value', 'top')
    storage_signature = 'std_bboxes'
    def __init__(self, prealloc_obj = 0):
        '''Initializes empty annotation, i.e. with no bounding boxes stored.'''
        self.value = ndarray((4, prealloc_obj), dtype = 'int32', order = 'C')
        self.top = 0
    def add_record(self, values):
        if self.top < self.value.shape[1]:
            self.value[:, self.top] = values
        else:
            self.value = hstack((self.value, column_stack((values,))))
        self.top += 1

class BoundingBoxLoader:
    annotation_class = BoundingBoxSampleAnnotation
    def setup_from_fields(self, field_names):
        en_names = enumerate(field_names)
        targets = dict([e[::-1] for e in en_names if e[1].startswith('bbox_')])
        if len(targets) == 0:
            raise NonApplicableLoader()
        all_cooords_error = None
        try:
            # trying to apply all-coordinates mode
            return self._setup_from_all_coords(targets)
        except (NonApplicableLoader, LoaderMissingData) as err:
            all_cooords_error = err
        # all-coordinates mode is failed at this point
        pin_and_size_error = None
        try:
            # trying to apply pin/size mode
            return self._setup_from_pin_and_size(targets)
        except (NonApplicableLoader, LoaderMissingData) as err:
            pin_and_size_error = err
        # both modes have failed at this point, we are going to provide
        # error description which include information about both problems
        raise RuntimeError('FIXME: both bbox loader modes have failed')
    def _setup_from_all_coords(self, targets_dict):
        all_coords = ('bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max')
        present_fields = set(all_coords) & targets_dict.keys()
        if len(present_fields) == 0:
            raise NonApplicableLoader()
        if len(present_fields) < len(all_coords):
            raise LoaderMissingData()
        def adapter(fields):
            return [int(e) for e in fields]
        # pick indices we use in a fixed order, return adapter function
        indices = [targets_dict[e] for e in all_coords]
        return indices, adapter
    def _setup_from_pin_and_size(self, targets_dict):
        pin_and_size = ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')
        present_fields = set(pin_and_size) & targets_dict.keys()
        if len(present_fields) == 0:
            raise NonApplicableLoader()
        if len(present_fields) < len(pin_and_size):
            raise LoaderMissingData()
        # pick needed indices we are going to process
        indices = [targets_dict[e] for e in pin_and_size]
        # for this format we need to do a little conversation (from side sizes
        # to coordinates of the other point)
        def adapter(fields):
            fields = [int32(e) for e in fields]
            fields[2] += fields[0]
            fields[3] += fields[1]
            return fields
        return indices, adapter
