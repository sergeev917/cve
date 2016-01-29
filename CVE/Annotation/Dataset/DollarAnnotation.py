# coding: utf8
'''Annotation.DollarAnnotation: class for Dollar's dataset format'''
__all__ = ('DollarAnnotation',)

from os.path import isdir, exists, splitext
from os import scandir
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

class DollarAnnotation(DatasetAnnotation):
    def __init__(self, path, **kwargs):
        # setting up an actual storage for samples
        self._storage = dict()
        if not exists(path):
            raise ValueError('The given path "{}" does not exist'.format(path))
        if not isdir(path):
            raise UnrecognizedAnnotationFormat('A directory is expected')
        directories_found = False
        # the version of Dollar format to use (differ in the number of fields)
        used_version = '0'
        # after the first version is specified or default version is used
        # the format version is fixed and must match any later specifications
        freeze_version = False
        format_version_line = '% bbGt version='
        for elem in scandir(path):
            if elem.name.startswith('.'):
                continue
            if elem.is_dir():
                directories_found = True
                continue
            # have a regular file here (or a symlink)
            markup_file = open(elem.path, 'r')
            line = markup_file.readline().strip()
            if not line.startswith(format_version_line):
                raise UnrecognizedAnnotationFormat(
                    'Dollar version header is missing for {}'.format(elem.name)
                )
            # cutting the beginning to get version string
            curr_version = line[len(format_version_line):]
            if freeze_version and curr_version != used_version:
                raise ViolatedAnnotationFormat(
                    'different format versions specifications are found'
                )
            if not freeze_version:
                used_version = curr_version
                freeze_version = True
                # since version is known at this point, it is time to construct
                # annotation generator to parse the rest of the file
                handlers_names = (
                    'ignore', # class name
                    'bbox_x',
                    'bbox_y',
                    'bbox_w',
                    'bbox_h',
                    'ignore', # occluded flag
                    'ignore', # visible bbox_x
                    'ignore', # visible bbox_y
                    'ignore', # visible bbox_w
                    'ignore', # visible bbox_h
                    'blacklist_01',
                    'ignore', # angle
                )
                fields_for_version = {'0': 10, '1': 10, '2': 11, '3': 12}
                if used_version not in fields_for_version:
                    raise ViolatedAnnotationFormat(
                        'version {} is not supported'.format(used_version)
                    )
                fields = handlers_names[:fields_for_version[used_version]]
                store = kwargs.get('adapter_store', get_global_registry())
                try:
                    adapters = set(map(lambda f: store[f], fields))
                except KeyError as error:
                    msg = 'Don\'t know how to process "{}" field'.format(error)
                    raise RuntimeError(msg) from None
                # building adapters from our field list (order of fields)
                adapters = tuple(map(lambda cls: cls(fields), adapters))
                # building storage-class with all declared annotations:
                # sub-annotation will be an attribute with predefined name
                self.storage_class = generate_sample_annotation_class(adapters)
            if directories_found:
                # dollar format is confirmed to be used here and directories
                # are not allowed (since we won't do a recursive scan)
                raise ViolatedAnnotationFormat(
                    'Dollar format prohibits subdirectories'
                )
            # now, version is fixed, reading annotation lines
            markup_obj = self.storage_class()
            for line in markup_file:
                field_values = tuple(map(str.strip, line.split()))
                markup_obj.push_all(*field_values)
            markup_file.close()
            # assigning name of the sample to be put into the storage
            sample_name = splitext(elem.name)[0]
            self._storage[sample_name] = markup_obj
    def __iter__(self):
        # items() returns dictionary view, which could be iterated over
        return self._storage.items().__iter__()
    def __getitem__(self, sample_name):
        # getting item from underlying dictionary storage
        return self._storage.__getitem__(sample_name)
