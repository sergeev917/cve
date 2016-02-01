# coding: utf8
__all__ = ('Plotter2D',)

from .Plugin import IPlugin
from matplotlib import use
from matplotlib.pyplot import figure, axes
from matplotlib.font_manager import FontProperties

# initializing toolset for matplotlib (`AGG` requires no running xorg)
use('AGG')

# FIXME: allow standalone workflow to plot multiple curves

class Plotter2D(IPlugin):
    '''Produces an image with a plotted curve of a function'''
    def __init__(self, filename, **opts):
        # making a deep copy to prevent mutability issues
        self.config = dict(opts)
        # preprocessing some reusable configuration
        if 'font' not in opts:
            fontname = opts.get('fontname', 'PT Sans Narrow')
            fontsize = opts.get('fontsize', 14)
            font = FontProperties(family = fontname, size = fontsize)
            self.config['font'] = font
        self.config['fileout'] = filename
    def inject(self, evaluator):
        # injecting `do_plot` as the last step in job queue
        evaluator.queue.append((do_plot, [], self.config))

def do_plot(workspace, **opts):
    # searching for data to plot
    output = workspace['out:driver']
    x, y = map(output.__getitem__, ('x-points', 'y-points'))
    x_entity, y_entity = map(output.__getitem__, ('x-entity', 'y-entity'))
    # setting up configuration
    font = opts['font']
    fileout = opts['fileout']
    language = opts.get('lang', 'en')
    axis_labels = {
        'en': {
            'precision': 'precision, %',
            'recall': 'recall, %',
            'false-positives': 'overall false positives',
            'false-positives-per-sample': 'mean false positives per image',
        },
        'ru': {
            'precision': 'точность, %',
            'recall': 'полнота, %',
            'false-positives': 'всего ложноположительных',
            'false-positives-per-sample': 'ложноположительных на изображение',
        },
    }[language]
    # setting axis labels
    x_axis_label = opts.get('x_axis_label', None)
    if x_axis_label == None:
        x_axis_label = axis_labels.get(x_entity, None)
        if x_axis_label == None:
            raise Exception(
                'unknown entity "{}", use x_axis_label option'.format(x_entity)
            )
    y_axis_label = opts.get('y_axis_label', None)
    if y_axis_label == None:
        y_axis_label = axis_labels.get(y_entity, None)
        if y_axis_label == None:
            raise Exception(
                'unknown entity "{}", use y_axis_label option'.format(y_entity)
            )
    # we will render precision and recall values as percentages
    x_percents, y_percents = map(
        ('precision', 'recall').__contains__,
        (x_entity, y_entity)
    )
    # we will render fp with log scale
    x_logscale, y_logscale = map(
        ('false-positives',).__contains__,
        (x_entity, y_entity)
    )
    # actual pyplot invokation starts here
    f = figure(figsize = (16, 16), dpi = 96)
    g = f.add_axes(axes())
    if x_percents:
        x = x * 100.
        g.set_xlim([-1, 101])
        g.set_xticks(range(0, 101, 5), minor = False)
        g.set_xticks(range(0, 101, 1), minor = True)
    if y_percents:
        y = y * 100.
        g.set_ylim([-1, 101])
        g.set_yticks(range(0, 101, 5), minor = False)
        g.set_yticks(range(0, 101, 1), minor = True)
    if x_logscale:
        g.set_xscale('log')
    if y_logscale:
        g.set_yscale('log')
    # plotting
    g.set_xlabel(x_axis_label, fontproperties = font)
    g.set_ylabel(y_axis_label, fontproperties = font)
    g.grid(which = 'major', alpha = 0.9)
    g.grid(which = 'minor', alpha = 0.2)
    g.plot(x, y, '-', aa = True, alpha = 0.7, label = opts.get('label', None))
    for label in g.get_xticklabels() + g.get_yticklabels():
        label.set_fontproperties(font)
    if 'label' in opts:
        g.legend(loc = 'lower left', prop = font)
    f.savefig(fileout, bbox_inches = 'tight')
