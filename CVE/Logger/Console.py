__all__ = (
    'ConsoleLog',
)

from shutil import get_terminal_size
from textwrap import shorten, wrap
from traceback import format_tb
from time import time
from math import trunc
from itertools import dropwhile, islice

from sys import stdout

color_scheme = (
    '',
    '\033[m',
    '\033[38;05;237m',
    '\033[01;38;05;196m',
    '\033[01;38;05;70m',
    '\033[38;05;172m',
)
def wrap_color(idx, text):
    return color_scheme[idx] + text + color_scheme[1]

def format_time_delta(seconds):
    full_seconds = trunc(seconds)
    ms = trunc(1000 * (seconds - full_seconds))
    minutes, seconds = divmod(full_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    values = [days, hours, minutes, seconds, ms]
    labels = ['day', 'hour', 'minute', 'second', 'millisecond']
    pair = islice(dropwhile(lambda e: e[0] == 0, zip(values, labels)), 0, 2)
    pair = list(filter(lambda e: e[0] != 0, pair))
    if len(pair) == 0:
        return 'less than a millisecond'
    pair = map(lambda e: (e[0], e[1] if e[0] <= 1 else e[1] + 's'), pair)
    return ' and '.join(map(lambda e: '{} {}'.format(*e), pair))

class LineManager:
    def __init__(self, width, file):
        self.use_width = width
        self.def_width = width
        self._file = file
        self._spacer = ''
        self._stack = []
        self._spacer_needed = True
    def push(self, prefix, length = None):
        if length == None:
            length = len(prefix)
        assert(self.use_width >= length)
        self.append(prefix, length)
        self._stack.append(len(self._spacer))
        self._spacer += ' ' * length
        self.def_width -= length
    def pop(self, count = 1):
        prev_spacer_size = len(self._spacer)
        curr_spacer_size = self._stack[-count]
        self._stack = self._stack[:-count]
        self.def_width += (prev_spacer_size - curr_spacer_size)
        self.use_width = self.def_width
        self._spacer = ' ' * curr_spacer_size
    def append(self, text, length = None, **kwargs):
        newline = kwargs.get('newline', False)
        flushright = kwargs.get('flushright', False)
        if length == None:
            length = len(text)
        assert(self.use_width >= length)
        if self._spacer_needed:
            self._spacer_needed = False
            text = self._spacer + text
        if flushright:
            text = ' ' * (self.use_width - length) + text
            length = self.use_width
        if newline:
            text += '\n'
            self._spacer_needed = True
            self.use_width = self.def_width
        else:
            self.use_width -= length
        self._file.write(text)
        self._file.flush()
    def reprint(self, text, length = None, **kwargs):
        newline = kwargs.get('newline', False)
        if length == None:
            length = len(text)
        text = '\r' + self._spacer + text
        if newline:
            self.use_width = self.def_width
            text += '\n'
        else:
            self.use_width = 0
        self._file.write(text)
        self._file.flush()

class SingleSubtask:
    reserve_width = 9
    def __init__(self, parent, descline, **kwargs):
        self.line = parent._line
        self.descline = descline
        self.use_timer = kwargs.get('timer', True)
    def __enter__(self):
        self.line.push('@ ')
        space = self.line.use_width - self.__class__.reserve_width
        line = shorten(self.descline, width = space, placeholder = '...')
        line += wrap_color(2, '.' * (space - len(line)))
        self.line.append(line, space)
        if self.use_timer:
            self._enter_time = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_timer:
            timer_msg = '(' + format_time_delta(time() - self._enter_time) + ')'
            colored_timer = wrap_color(5, timer_msg), len(timer_msg)
        if exc_type == None:
            self.line.append(wrap_color(4, 'COMPLETED'), 9, newline = True)
            self.line.pop()
            self.line.append(*colored_timer, newline = True, flushright = True)
            return
        self.line.append(wrap_color(3, 'SHATTERED'), 9, newline = True)
        # calculating current offset
        self.line.push(wrap_color(3, '[ERROR] '), 8)
        msg = 'Traceback:\n' + '\n'.join(format_tb(traceback))
        msg += '\n{}: {}'.format(exc_type.__name__, exc_value)
        wrap_line = lambda l: wrap(l, width = self.line.def_width)
        for lines in map(wrap_line, msg.split('\n')):
            for line in lines:
                self.line.append(line, newline = True)
        self.line.newline()
        self.line.pop()
        raise Exception('Subtask has failed') from None

class SpinnerSubtask:
    def __init__(self, parent, descline, ticks = 100):
        self.line = parent._line
        self.descline = descline
        self._fullticks = ticks
        self._fill = ('|', '\u00b7')
    def __enter__(self):
        self.line.push('@ ')
        self._enter_time = time()
        self._currticks = 0
        self._filled_dots = 0
        space = self.line.use_width
        line = shorten(self.descline, width = space, placeholder = '...')
        self.line.append(line, newline = True)
        self._bar_size = self.line.use_width - 7
        return self
    def tick(self, count = 1):
        self._currticks += count
        self.__bar()
    def __bar(self):
        dots = trunc(self._bar_size * self._currticks / self._fullticks)
        dots = min(dots, self._bar_size)
        if self._filled_dots != dots:
            self._filled_dots = dots
            percents = trunc(100. * self._currticks / self._fullticks)
            bar = wrap_color(4, self._fill[0] * dots)
            bar += wrap_color(2, self._fill[1] * (self._bar_size - dots))
            bar = '%03d%% [%s]' % (percents, bar)
            self.line.reprint(bar, self.line.def_width)
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == None:
            timer_msg = '(' + format_time_delta(time() - self._enter_time) + ')'
            timer_len = len(timer_msg)
            timer_msg = wrap_color(5, timer_msg)
            line = ' ' * (self.line.def_width - timer_len) + timer_msg
            self.line.reprint(line, self.line.use_width, newline = True)
            self.line.pop()
            return
        self.line.append('', newline = True)
        self.line.push(wrap_color(3, '[ERROR] '), 8)
        msg = 'Traceback:\n' + '\n'.join(format_tb(traceback))
        msg += '\n{}: {}'.format(exc_type.__name__, exc_value)
        wrap_line = lambda l: wrap(l, width = self.line.def_width)
        for lines in map(wrap_line, msg.split('\n')):
            for line in lines:
                self.line.append(line, newline = True)
        self.line.pop()
        #raise Exception('Subtask has failed') from None
        raise SystemExit(1)

class ConsoleLog:
    __dummy__ = False
    def __init__(self, file = stdout):
        term_sizes = get_terminal_size()
        self._line = LineManager(term_sizes.columns, file)
    def subtask(self, descline, **kwargs):
        return SingleSubtask(self, descline, **kwargs)
    def progress(self, descline, ticks):
        return SpinnerSubtask(self, descline, ticks)
    def info(self, message):
        print(message)
    def warn(self, message):
        print(message)
    def fail(self, message):
        print(message)
