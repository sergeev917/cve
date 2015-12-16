# coding: utf8
'''Annotation.SimpleDetectionsList: class for simple detections list format'''
__all__ = ('SimpleDetectionsList',)

from os.path import isfile, exists
from itertools import chain
from ..Util import parse_format_header
from .Common import (
    UnrecognizedAnnotationFormat,
    ViolatedAnnotationFormat,
    DatasetAnnotation,
)

class SimpleDetectionsList(DatasetAnnotation):
    signature_name = 'simple detections list'

    def __init__(self, path):
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
        # constructing a class for annotations based on the given parameters
        # NOTE: this class type is unique per generator invokation

        # setting up actual storage for samples
        self._storage = dict()
        def annotation_init(sample_name):
            if sample_name not in self._storage:
                self._storage[sample_name] = SimpleDetectionsListAnnotation()
            return self._storage[sample_name]


        def process_line(line):
            fields = map(str.strip, line.split(separator))
        # reading detections, one detections per line
        for line in chain(pending_line, markup_file):
            process_line(line)
