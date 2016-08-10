# coding: utf8
'''Annotation.SimpleDetectionsList: class for simple detections list format'''
__all__ = ('SimpleDetectionsList',)

from os.path import isfile, exists, basename, splitext
from itertools import chain
from operator import itemgetter
from ..Sample import compose_annotation_class
from ...Logger import get_default_logger
from ..Util import (
    parse_format_header,
    get_global_registry,
    count_lines,
)
from ...Base import (
    UnrecognizedAnnotationFormat,
    ViolatedAnnotationFormat,
    IDatasetAnnotation,
)

class SimpleDetectionsList(IDatasetAnnotation):
    signature_name = 'simple detections list'
    def __init__(self, path, **kwargs):
        # FIXME: +docstring adapter_store in kwargs
        progress = kwargs.get('progress', True)
        if progress:
            logger = kwargs.get('logger', get_default_logger())
            # disable progress-bar overhead if the current logger is a devnull
            if logger.__dummy__:
                progress = False
        if not exists(path):
            raise ValueError('The given path "{}" does not exist'.format(path))
        if not isfile(path):
            raise UnrecognizedAnnotationFormat('A regular file is expected')
        # if the progress bar is requested we need to count lines in the markup
        if progress:
            msg = 'reading ground-truth files in "{}"'.format(path)
            with logger.progress(msg, count_lines(path)) as spinner:
                self.__load__(path, spinner.tick, **kwargs)
        else:
            self.__load__(path, **kwargs)
    def __load__(self, path, tick = None, **kwargs):
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
        tick() if tick else None # tick for `signature_line` reading
        for buffered_line in markup_file:
            if not buffered_line.startswith('#'):
                # we've already read it, but it is not from header
                pending_line = [buffered_line]
                break
            # stripping comment and newline symbols
            format_header_lines.append(buffered_line[1:].strip())
            tick() if tick else None # tick for the appended line
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
        storage_class = compose_annotation_class(fields, adapter_store)
        self.storage_class = storage_class
        # setting up actual storage for samples
        self._storage = dict()
        def access(sample_name):
            if sample_name not in self._storage:
                self._storage[sample_name] = storage_class(0)
            return self._storage[sample_name]
        # because it is highly possible that several lines of the data will be
        # about the same sample (multiple detections on a single image/sample)
        # we will buffer the last sample annotation to avoid looking it up
        buffered_sample_annotation = (None, None)
        # reading detections, one detections per line
        for line in chain(pending_line, markup_file):
            field_values = [e.strip() for e in line.split(separator)]
            sample_name = get_sample_name(field_values)
            if sample_name != buffered_sample_annotation[0]:
                buffered_sample_annotation = (sample_name, access(sample_name))
            buffered_sample_annotation[1].add_record(field_values)
            tick and tick() # tick for the current line
    def __iter__(self):
        # items() returns dictionary view, which could be iterated over
        return self._storage.items().__iter__()
    def __getitem__(self, sample_name):
        # getting item from underlying dictionary storage
        # NOTE: this format doesn't keep records of samples with no detections,
        #       so any unseen sample name will produce empty annotation here
        annotation = self._storage.get(sample_name, None)
        if annotation == None:
            return self.storage_class(0)
        return annotation
    def __len__(self):
        return self._storage.__len__()
