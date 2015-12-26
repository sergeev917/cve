__all__ = (
    'BoundingBoxSampleAnnotation',
    'BoundingBoxFieldAdapter',
)

from operator import itemgetter
from numpy import (
    ndarray,
    hstack,
    column_stack,
)

class BoundingBoxSampleAnnotation:
    '''The class stores single matrix which represents all bounding boxes on
       given sample (which is supposed to be a single image). The data layout
       is designed to accelerate matrix operations such as selecting max or
       min on one feature channel (for instance, x_min or y_max), not one
       bounding box. So note that the data is stored row-wise, a bounding box
       is a column in the matrix.'''
    __slots__ = ('value',)
    signature_name = 'numpy_bounding_boxes'

    def __init__(self):
        '''Initializes empty annotation, i.e. with no bounding boxes stored.'''
        self.value = ndarray((4, 0), dtype = 'int32', order = 'C')

    # NOTE: if the following function is changed then format parsing function
    # in the BoundingBoxFieldAdapter class should be corrected accordingly
    # (whether order of fields or meaning was altered).
    def push(self, x_min, y_min, x_max, y_max):
        new_bbox_column = column_stack(([x_min, y_min, x_max, y_max],))
        self.value = hstack((self.value, new_bbox_column))

class BoundingBoxFieldAdapter:
    storage_class = BoundingBoxSampleAnnotation
    handled_fields  = ('bbox_x_min', 'bbox_x_max', 'bbox_y_min', 'bbox_y_max',
                       'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')

    def create_annotation(self):
        return self.__class__.storage_class()

    def __init__(self, field_names):
        # selecting options which are related to bounding boxes
        bbox_options = set(filter(lambda s: s.startswith('bbox_'), field_names))
        if len(bbox_options) == 0:
            raise Exception() # FIXME
        # will choose the following function based on format (field names set):
        # target bbox format is x/y points (no width/height), so postproc_func
        # will be set for width-and-height format to summ up width/height with
        # the segments start points (for another format postproc_func will
        # remain to be None as no additional actions are required).
        pickup_func = None
        postproc_func = None
        format_is_supported = False
        # checking for x-min/max, y-min/max format of bbox
        required = ('bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max')
        present_options = set(required) & bbox_options
        if len(present_options) == len(required):
            # preparing a function that will pick required fields
            indices = map(field_names.index, required)
            pickup_func = itemgetter(*list(indices))
            format_is_supported = True
        elif len(present_options) != 0:
            raise Exception() # FIXME
        # checking for x, y, w, h format of bbox if no format found so far
        if not format_is_supported:
            required = ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')
            present_options = set(required) & bbox_options
            if len(present_options) == len(required):
                # preparing a function that will pick required fields
                indices = map(field_names.index, required)
                pickup_func = itemgetter(*list(indices))
                def postproc_func(integer_bboxes):
                    integer_bboxes[2] += integer_bboxes[0]
                    integer_bboxes[3] += integer_bboxes[1]
                format_is_supported = True
            elif len(present_options) != 0:
                raise Exception() # FIXME
        # checking whether a supported format was found
        if not format_is_supported:
            raise Exception() # FIXME
        # constructing a function which will append annotation
        # in the choosen format (from fields string-values)
        def append_annotation(annotation_object, *field_values):
            nonlocal pickup_func, postproc_func
            integer_bboxes = list(map(int, pickup_func(field_values)))
            # NOTE: check performance with the do-nothing lambda
            if postproc_func != None:
                postproc_func(integer_bboxes)
            annotation_object.push(*integer_bboxes)
        self.append_annotation = append_annotation
