# coding: utf8
__all__ = ('Plotter2D',)

from ..Base import IPlugin
from matplotlib import use
from matplotlib.pyplot import figure, axes
from matplotlib.font_manager import FontProperties

# initializing toolset for matplotlib (`AGG` requires no running xorg)
use('AGG')

# FIXME: allow standalone workflow to plot multiple curves

class Plotter2D(IPlugin):
    '''Produces an image with a plotted curve of a function'''
    def __init__(self, save_path, **opts):
        # making a deep copy to prevent mutability issues
        self.config = dict(opts)
        # preprocessing some reusable configuration
        if 'font' not in opts:
            fontname = opts.get('fontname', 'PT Sans Narrow')
            fontsize = opts.get('fontsize', 14)
            font = FontProperties(family = fontname, size = fontsize)
            self.config['font'] = font
        self.config['save_path'] = save_path
    def inject(self, evaluator):
        # injecting `do_plot` as the last step in job queue
        evaluator.queue.append((do_plot, [], self.config))

def do_plot(workspace, **options):
    font_obj = options['font']
    save_path = options['save_path']
    driver_data = workspace['export:driver']
    graph_label = options.get('label', None)
    if graph_label != None:
        auc = driver_data.get('auc', None)
        if auc != None:
            graph_label += ' (AP {:.3f})'.format(auc)
    # labels for axes: lookup for [xy]_axis_label in options, then lookup for
    # entity_labels with specified language, then use raw entity name
    axis_labels = entity_labels[options.get('lang', 'en')]
    def get_label_for(entity):
        label = axis_labels.get(entity, None)
        if label == None:
            return entity
        return label
    # expecting to find [xy]-points and [xy]-entity in the driver output
    x_entity, y_entity = map(driver_data.__getitem__, ('x-entity', 'y-entity'))
    x = apply_data_handlers(x_entity, driver_data['x-points'])
    y = apply_data_handlers(y_entity, driver_data['y-points'])
    x_axis_label = options.get('x_axis_label', get_label_for(x_entity))
    y_axis_label = options.get('y_axis_label', get_label_for(y_entity))
    # actual pyplot invokation starts here
    f = figure(figsize = (16, 16), dpi = 96)
    g = f.add_axes(axes())
    apply_view_handlers(x_entity, x, 'x', g)
    apply_view_handlers(y_entity, y, 'y', g)
    # plotting
    g.set_xlabel(x_axis_label, fontproperties = font_obj)
    g.set_ylabel(y_axis_label, fontproperties = font_obj)
    g.grid(which = 'major', alpha = 0.9)
    g.grid(which = 'minor', alpha = 0.2)
    g.plot(x, y, '-', aa = True, alpha = 0.7,
           **({} if not graph_label else {'label': graph_label}))
    for label in g.get_xticklabels() + g.get_yticklabels():
        label.set_fontproperties(font_obj)
    if 'label' in options:
        g.legend(loc = 'lower left', prop = font_obj)
    f.savefig(save_path, bbox_inches = 'tight')

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
