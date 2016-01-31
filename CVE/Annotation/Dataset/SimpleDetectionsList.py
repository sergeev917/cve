# coding: utf8
'''Annotation.SimpleDetectionsList: class for simple detections list format'''
__all__ = ('SimpleDetectionsList',)

from os.path import isfile, exists, basename, splitext
from itertools import chain
from operator import itemgetter
from ..Sample import generate_sample_annotation_class
from ..Util import (
    parse_format_header,
    get_global_registry,
)
from .DatasetAnnotation import (
    UnrecognizedAnnotationFormat,
    ViolatedAnnotationFormat,
    DatasetAnnotation,
)

class SimpleDetectionsList(DatasetAnnotation):
    signature_name = 'simple detections list'

    def __init__(self, path, **kwargs):
        # FIXME: +docstring adapter_store in kwargs
        if not exists(path):
            raise ValueError('The given path "{}" does not exist'.format(path))
        if not isfile(path):
            raise UnrecognizedAnnotationFormat('A regular file is expected')
        markup_file = open(path, 'r')
        # now reading the first line where a signature should be located
        signature_line = markup_file.readline().strip()
        if not signature_line.startswith('#'):
            raise UnrecognizedAnnotationFormat('The signature line is missing')
        signature_line = signature_line[1:].strip()
        # checking format name presence in the format string
        if signature_line != self.__class__.signature_name:
            raise UnrecognizedAnnotationFormat('No signature match')
        # now checking for header continuation with format-specific options
        format_header_lines = []
        pending_line = []
        for buffered_line in markup_file:
            if not buffered_line.startswith('#'):
                # we've already read it, but it is not from header
                pending_line = [buffered_line]
                break
            # stripping comment and newline symbols
            format_header_lines.append(buffered_line[1:].strip())
        # don't forget that there is a line in buffer (pending_line)
        options = parse_format_header(' '.join(format_header_lines))
        # loading the required configuration options
        try:
            separator = options['separator']
            fields = options['fields']
        except KeyError as error:
            msg = '"{}" option is required to be set'.format(error)
            raise ViolatedAnnotationFormat(msg) from None
        # checking for sample-name fields because otherwise we can't identify
        # any particular sample; also we're replacing "name" with
        # "ignore" because we're dealing with name-field by ourselves
        sample_name_fields = ('name', 'path')
        present = tuple(filter(lambda f: f in sample_name_fields, fields))
        if len(present) == 0:
            raise ViolatedAnnotationFormat(
                '"name" or "path" must be set to distinguish samples'
            )
        if len(present) > 1:
            raise ViolatedAnnotationFormat(
                'only one sample name-related field must be specified'
            )
        selected_name_type = present[0]
        sample_name_index = fields.index(selected_name_type)
        fields[sample_name_index] = 'ignore'
        get_name_field = itemgetter(sample_name_index)
        if selected_name_type == 'name':
            # a sample name is actually just value of the selected field
            get_sample_name = get_name_field
        elif selected_name_type == 'path':
            # a sample name is a path, need to extract the file name
            def get_sample_name(fields):
                return splitext(basename(get_name_field(fields)))[0]
        # getting a storage to get adapter classes from (using field names)
        adapter_store = kwargs.get('adapter_store', get_global_registry())
        try:
            adapters = set(map(lambda f: adapter_store[f], fields))
        except KeyError as error:
            msg = 'Don\'t know how to process "{}" field'.format(error)
            raise RuntimeError(msg) from None
        # building adapters from our field list (order of fields, specifically)
        adapters = tuple(map(lambda cls: cls(fields), adapters))
        # building storage-class which will contain all declared annotations
        # as attributes with predefined names
        self.storage_class = generate_sample_annotation_class(adapters)
        storage_class = self.storage_class
        # setting up actual storage for samples
        self._storage = dict()
        def access(sample_name):
            if sample_name not in self._storage:
                self._storage[sample_name] = storage_class()
            return self._storage[sample_name]
        # because it is highly possible that several lines of the data will be
        # about the same sample (multiple detections on a single image/sample)
        # we will buffer the last sample annotation to avoid looking it up
        buffered_sample_annotation = (None, None)
        # reading detections, one detections per line
        for line in chain(pending_line, markup_file):
            field_values = tuple(map(str.strip, line.split(separator)))
            sample_name = get_sample_name(field_values)
            if sample_name != buffered_sample_annotation[0]:
                buffered_sample_annotation = (sample_name, access(sample_name))
            buffered_sample_annotation[1].push_all(*field_values)

    def __iter__(self):
        # items() returns dictionary view, which could be iterated over
        return self._storage.items().__iter__()

    def __getitem__(self, sample_name):
        # getting item from underlying dictionary storage
        # NOTE: this format doesn't keep records of samples with no detections,
        #       so any unseen sample name will produce empty annotation here
        annotation = self._storage.get(sample_name, None)
        if annotation == None:
            return self.storage_class()
        return annotation
