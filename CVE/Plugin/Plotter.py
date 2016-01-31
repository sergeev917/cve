__all__ = ('Plotter2D',)

from .Plugin import IPlugin
from matplotlib import use
from matplotlib.pyplot import figure, subplot, plot, legend, savefig, close
from matplotlib.font_manager import FontProperties

# initializing toolset for matplotlib (`AGG` requires no running xorg)
use('AGG')

# FIXME: allow standalone workflow to plot multiple curves

class Plotter2D(IPlugin):
    '''Produces an image with a plotted curve of a function'''
    def __init__(self, filename, **opts):
        fontname = opts.get('fontname', 'PT Sans Narrow')
        fontsize = opts.get('fontsize', 14)
        self.font = FontProperties(family = fontname, size = fontsize)
        self.filename = filename
    def inject(self, evaluator):
        def do_plot(workspace):
            # searching for data to plot
            output = workspace['out:driver']
            x = output['x-points'] * 100.
            y = output['y-points'] * 100.
            # plotting
            figure(figsize = (20, 20), dpi = 101)
            axes = subplot()
            axes.set_xlim([-1, 101])
            axes.set_ylim([-1, 101])
            axes.set_xlabel('Recall, %', fontproperties = self.font)
            axes.set_ylabel('Precision, %', fontproperties = self.font)
            axes.set_aspect('equal')
            axes.set_xticks(range(0, 101, 5), minor = False)
            axes.set_xticks(range(0, 101, 1), minor = True)
            axes.set_yticks(range(0, 101, 5), minor = False)
            axes.set_yticks(range(0, 101, 1), minor = True)
            axes.grid(which = 'major', alpha = 0.9)
            axes.grid(which = 'minor', alpha = 0.2)
            plot(x, y, '-', aa = True, alpha = 0.7, label = 'current graph')
            for label in axes.get_xticklabels() + axes.get_yticklabels():
                label.set_fontproperties(self.font)
            legend(loc = 'lower left', prop = self.font)
            savefig(self.filename, bbox_inches = 'tight')
            close()
        # injecting `do_plot` as the last step in job queue
        evaluator.queue.append((do_plot, [], {}))
