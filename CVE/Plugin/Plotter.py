# coding: utf8
__all__ = (
    'CurvePlotter2D',
)

from re import match
from re import compile as _compile
from ..Logger import get_default_logger
from matplotlib import use
from matplotlib.pyplot import figure, axes
from matplotlib.font_manager import FontProperties

# initializing toolset for matplotlib (`AGG` requires no running xorg)
use('AGG')

# FIXME: allow standalone workflow to plot multiple curves

class CurvePlotter2D:
    '''Produces an image with a plotted curve of a function'''
    _contract_provide = ('__curve_plotter_side_effect__',)
    _curve_pattern = _compile('^analysis:(?P<problem_class>[^:]+):' \
                              'curve-(?P<y_entity>[^(]+)\((?P<x_entity>[^)]+)\)$')
    def __init__(self, image_out_path, **opts):
        self._logger = opts.get('logger', get_default_logger())
        # making a deep copy to prevent mutability issues
        self.config = dict(opts)
        self.config['out_path'] = image_out_path
        # preprocessing some reusable configuration
        if 'font' not in opts:
            self.config['font'] = FontProperties(
                family = opts.get('fontname', 'PT Sans Narrow'),
                size = opts.get('fontsize', 14),
            )
        # dynamic contracts persistence data
        self._input_contract_mapper = {}
        self._modes_metainfo = []
    def targets_to_inject(self):
        return self._contract_provide
    def dynamic_contracts(self, target_name, available_resources):
        # fulfill only our injected target
        if target_name != self._contract_provide[0]:
            return []
        # search what we are going to plot: resources named like
        # 'analysis:...:curve-...(...)'
        curve_data, supp_data = [], []
        for resource_name in available_resources:
            match_data = match(self._curve_pattern, resource_name)
            if match_data is not None:
                fields = match_data.groupdict()
                curve_data.append((resource_name, fields))
            if resource_name == 'analysis:object-detection:mean-avg-precision':
                supp_data.append(resource_name)
        if len(curve_data) == 0:
            return []
        curve_data = curve_data[0]
        # requesting the resource with a curve and supplementary information
        require_list = (curve_data[0],) + tuple(supp_data)
        mode_id = self._input_contract_mapper.get(require_list, None)
        if mode_id is not None:
            # we already assigned mode id for the current input set
            return [mode_id]
        mode_id = len(self._modes_metainfo) # not thread-safe (race)
        self._input_contract_mapper[require_list] = mode_id
        ents = tuple(map(curve_data[1].__getitem__, ['y_entity', 'x_entity']))
        self._modes_metainfo.append((require_list, ents, supp_data))
        return [mode_id]
    def get_contract(self, mode_id):
        return (self._modes_metainfo[mode_id][0], self._contract_provide)
    def setup(self, mode_id, input_types, output_mask):
        entities, supp_names = self._modes_metainfo[mode_id][1:]
        y_entity, x_entity = entities
        options = self.config
        try:
            map_index = \
                supp_names.index('analysis:object-detection:mean-avg-precision')
        except ValueError:
            map_index = None
        # for now we won't be doing anything flexible about supplementary
        # resources and confine ourselves to only mAP information
        def worker_func(curve_data, *supp_data):
            font_prop_obj = options['font']
            img_out_path = options['out_path']
            graph_label = options.get('label', None)
            if graph_label is not None and map_index is not None:
                graph_label += ' (mAP {:.3f})'.format(supp_data[map_index].auc)
            # labels for axes: lookup for [xy]_axis_label in options, then lookup for
            # entity_labels with specified language, then use raw entity name
            axis_labels = entity_labels[options.get('lang', 'en')]
            def get_label_for(entity):
                label = axis_labels.get(entity, None)
                if label == None:
                    return entity
                return label
            # expecting to find [xy]-points and [xy]-entity in the driver output
            x = apply_data_handlers(x_entity, curve_data.x_points)
            y = apply_data_handlers(y_entity, curve_data.y_points)
            x_axis_label = options.get('x_axis_label', get_label_for(x_entity))
            y_axis_label = options.get('y_axis_label', get_label_for(y_entity))
            # actual pyplot invokation starts here
            f = figure(figsize = (16, 16), dpi = 96)
            g = f.add_axes(axes())
            apply_view_handlers(x_entity, x, 'x', g)
            apply_view_handlers(y_entity, y, 'y', g)
            # plotting
            g.set_xlabel(x_axis_label, fontproperties = font_prop_obj)
            g.set_ylabel(y_axis_label, fontproperties = font_prop_obj)
            g.grid(which = 'major', alpha = 0.9)
            g.grid(which = 'minor', alpha = 0.2)
            g.plot(x, y, '-', aa = True, alpha = 0.7,
                   **({} if not graph_label else {'label': graph_label}))
            for label in g.get_xticklabels() + g.get_yticklabels():
                label.set_fontproperties(font_prop_obj)
            if 'label' in options:
                g.legend(loc = 'lower left', prop = font_prop_obj)
            f.savefig(img_out_path, bbox_inches = 'tight')
        if self._logger.__dummy__:
            return worker_func, (None,)
        # need to make a logger wrapped worker
        def log_wrapped_worker(*args, **kwargs):
            msg = 'plotting the curve (output is at "{}")'.format(
                self.config['out_path'],
            )
            with self._logger.subtask(msg):
                return worker_func(*args, **kwargs)
        return log_wrapped_worker, (None,)

# labels to be plotted on a graph to describe what axis shows
entity_labels = {
    'en': {
        'precision': 'precision, %',
        'recall': 'recall, %',
        'false-positives': 'overall false positives',
        'false-positives-per-sample': 'mean false positives per image',
    },
    'ru': {
        'precision': 'точность, %',
        'recall': 'полнота, %',
        'false-positives': 'всего ложных срабатываний',
        'false-positives-per-sample': 'ложных срабатываний на изображение',
    },
}

def do_axes_commands(view_processor_func):
    def wrapped(data, axes, axis_name):
        cmds = view_processor_func(data, axes, axis_name)
        for option, args, kwrgs in cmds:
            func = getattr(axes, 'set_{}{}'.format(axis_name, option))
            func(*args, **kwrgs)
    return wrapped

@do_axes_commands
def percent_view(data, axes, axis_name):
    return [
        ('lim', (-1, 101), {}),
        ('ticks', (range(0, 101, 5),), {'minor': False}),
        ('ticks', (range(0, 101, 1),), {'minor': True}),
    ]

@do_axes_commands
def logscale_view(data, axes, axis_name):
    return [
        ('scale', ('log',), {}),
    ]

def percent_converter(data):
    return data * 100.

data_handlers = {
    'precision': [percent_converter],
    'recall': [percent_converter],
}

view_handlers = {
    'precision': [percent_view],
    'recall': [percent_view],
    'false-positives': [logscale_view],
    'false-positives-per-sample': [logscale_view],
}

def apply_data_handlers(entity, data):
    func_list = data_handlers.get(entity, [])
    for func in func_list:
        data = func(data)
    return data

def apply_view_handlers(entity, data, axis_name, axes):
    func_list = view_handlers.get(entity, [])
    for func in func_list:
        func(data, axes, axis_name)
