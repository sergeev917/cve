# coding: utf8
'''Annotation.DollarAnnotation: class for Dollar's dataset format'''
__all__ = ('DollarAnnotation',)

from os.path import isdir, exists, splitext
from os import scandir
from itertools import chain
from operator import itemgetter
from ..Sample import compose_annotation_class
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
from ...Logger import get_default_logger

# FIXME: suppress output until we're sure that we're dealing with a dollar base
class DollarAnnotation(IDatasetAnnotation):
    def __init__(self, path, **kwargs):
        progress = kwargs.get('progress', True)
        if progress:
            logger = kwargs.get('logger', get_default_logger())
            # disable progress-bar overhead if the current logger is a devnull
            if logger.__dummy__:
                progress = False
        if not exists(path):
            raise ValueError('The given path "{}" does not exist'.format(path))
        if not isdir(path):
            raise UnrecognizedAnnotationFormat('A directory is expected')
        if not progress:
            self.__load__(path, **kwargs)
        else:
            # user has requested a progress bar, so we need to count files
            file_count = 0
            for elem in scandir(path):
                if not elem.name.startswith('.') and not elem.is_dir():
                    file_count += 1
            msg = 'reading ground-truth files in "{}"'.format(path)
            with logger.progress(msg, file_count) as spinner:
                self.__load__(path, spinner.tick, **kwargs)
    def __load__(self, path, tick = None, **kwargs):
        directories_found = False
        # the version of Dollar format to use (differ in the number of fields)
        used_version = '0'
        # after the first version is specified or default version is used
        # the format version is fixed and must match any later specifications
        freeze_version = False
        format_version_line = '% bbGt version='
        # setting up an actual storage for samples
        self._storage = dict()
        for elem in scandir(path):
            if elem.name.startswith('.'):
                continue
            if elem.is_dir():
                directories_found = True
                continue
            # have a regular file here (or a symlink)
            markup_lines_count = count_lines(elem.path) - 1
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
                    'whitelist_01',
                    'ignore', # angle
                )
                fields_for_version = {'0': 10, '1': 10, '2': 11, '3': 12}
                if used_version not in fields_for_version:
                    raise ViolatedAnnotationFormat(
                        'version {} is not supported'.format(used_version)
                    )
                fields = handlers_names[:fields_for_version[used_version]]
                store = kwargs.get('adapter_store', get_global_registry())
                self.storage_class = compose_annotation_class(fields, store)
            if directories_found:
                # dollar format is confirmed to be used here and directories
                # are not allowed (since we won't do a recursive scan)
                raise ViolatedAnnotationFormat(
                    'Dollar format prohibits subdirectories'
                )
            # now, version is fixed, reading annotation lines
            markup_obj = self.storage_class(markup_lines_count)
            for line in markup_file:
                markup_obj.add_record(
                    [e.strip() for e in line.split()],
                )
            markup_file.close()
            # assigning name of the sample to be put into the storage
            sample_name = splitext(elem.name)[0]
            self._storage[sample_name] = markup_obj
            # notifying the progress bar to tick forward (if any)
            tick and tick()
    def __iter__(self):
        # items() returns dictionary view, which could be iterated over
        return self._storage.items().__iter__()
    def __getitem__(self, sample_name):
        # getting item from underlying dictionary storage
        return self._storage.__getitem__(sample_name)
    def __len__(self):
        return self._storage.__len__()
