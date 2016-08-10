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
    '\033[m', # reset color
    '\033[38;05;237m', # grey for fillers
    '\033[01;38;05;196m', # red, errors
    '\033[01;38;05;70m', # green, good
    '\033[38;05;172m', # yellowish, timer
    '\033[01;38;05;62m', # blue, information
    '\033[01;38;05;214m', # orange, warnings
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
    '''Controls line printing and prefixes stacking'''
    def __init__(self, width, file):
        self.use_width = width
        self.def_width = width
        self._file = file
        self._spacer = ''
        self._stack = []
        self._spacer_needed = True
    def push(self, prefix, spacer = None, length = None):
        if length is None:
            length = len(prefix)
        if spacer is None:
            spacer = ' ' * length
        assert(self.use_width >= length)
        self.append(prefix, length)
        self._stack.append(len(self._spacer))
        self._spacer += spacer
        self.def_width -= length
    def pop(self, count = 1):
        prev_spacer_size = len(self._spacer)
        curr_spacer_size = self._stack[-count]
        self._stack = self._stack[:-count]
        self.def_width += (prev_spacer_size - curr_spacer_size)
        self.use_width = self.def_width
        self._spacer = self._spacer[:curr_spacer_size]
    def message(self, message):
        assert self.use_width > 0
        wrap_line = lambda l: wrap(l, width = self.use_width)
        for lines in map(wrap_line, message.split('\n')):
            for line in lines:
                self.append(line, newline = True)
    def append(self, text, length = None, **kwargs):
        newline = kwargs.get('newline', False)
        if length == None:
            length = len(text)
        assert(self.use_width >= length)
        if self._spacer_needed:
            self._spacer_needed = False
            text = self._spacer + text
        if newline:
            text += '\n'
            self._spacer_needed = True
            self.use_width = self.def_width
        else:
            self.use_width -= length
        self._file.write(text)
        self._file.flush()
    def reprint(self, text, **kwargs):
        newline = kwargs.get('newline', False)
        text = '\r' + self._spacer + text
        if newline:
            self.use_width = self.def_width
            text += '\n'
        else:
            self.use_width = 0
        self._file.write(text)
        self._file.flush()

class SingleSubtask:
    def __init__(self, parent, description_line, **kwargs):
        '''Initializes a new subtask with given description'''
        self._parent = parent
        self._lineman = parent._lineman
        self._description = description_line
        self._use_timer = kwargs.get('timer', parent._use_timer)
    def __enter__(self):
        '''Printing the subtask description and starts timer if needed'''
        # reverse space for okay/fail label on the right
        space = self._lineman.use_width - 4
        line = shorten(self._description, width = space, placeholder = '...')
        # filling the space between the line end and the label with dots
        line += wrap_color(2, '.' * (space - len(line)))
        self._lineman.append(line, space)
        # whether we need to print label on the right
        # will be disabled when subtasks pushed
        self._need_status_label = True
        if self._use_timer:
            self._enter_time = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if self._use_timer:
            timer  = ' - took ' + format_time_delta(time() - self._enter_time)
            color_time = wrap_color(5, timer), len(timer)
        # printing okay/fail and additional info if needed
        if exc_type is None:
            if self._need_status_label:
                self._lineman.append(wrap_color(4, 'okay'), 4, newline = True)
            if self._use_timer:
                self._lineman.append(*color_time, newline = True)
            return
        # failure case
        if self._need_status_label:
            self._lineman.append(wrap_color(3, ' [!]'), 4, newline = True)
        self.fail('{}: {}'.format(exc_type.__name__, exc_value))
        raise
    def subtask(self, descline, **kwargs):
        # now we're displaying dotted line with pending COMPLETED/SHATTERED:
        # since now there is a subtask requested, we need to indicate that
        self._need_status_label = False
        self._lineman.append(wrap_color(4, '-**-'), 4, newline = True)
        return SingleSubtask(self, descline, **kwargs)
    def progress(self, descline, ticks):
        self._need_status_label = False
        self._lineman.append(wrap_color(4, '-**-'), 4, newline = True)
        return SpinnerSubtask(self, descline, ticks)
    def info(self, message):
        self._parent.info(message)
    def warn(self, message):
        self._parent.warn(message)
    def fail(self, message):
        self._parent.fail(message)

class SpinnerSubtask:
    def __init__(self, parent, description_line, ticks, **kwargs):
        '''Initializes a new loopy subtask with given description'''
        # percentage step to print after
        percentage_step = kwargs.get('period', 5)
        self._print_period = ticks * percentage_step // 100
        self._all_ticks = ticks
        self._parent = parent
        self._lineman = parent._lineman
        self._description = description_line
        self._use_timer = kwargs.get('timer', parent._use_timer)
    def __enter__(self):
        if self._use_timer:
            self._enter_time = time()
        self._curr_ticks = 0
        self._next_threshold = 0
        # reserving space for percentage
        space = self._lineman.use_width - 4
        line = shorten(self._description, width = space, placeholder = '...')
        line += wrap_color(2, '.' * (space - len(line)))
        self._prefix = line
        self._push_label(wrap_color(3, '  0%'))
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if self._use_timer:
            timer  = ' - took ' + format_time_delta(time() - self._enter_time)
            color_time = wrap_color(5, timer), len(timer)
        if exc_type is None:
            self._push_label(wrap_color(4, 'okay'))
            self._lineman.append('', newline = True)
            if self._use_timer:
                self._lineman.append(*color_time, newline = True)
            return
        self._push_label(wrap_color(3, ' [!]'))
        self._lineman.append('', newline = True)
        self.fail('{}: {}'.format(exc_type.__name__, exc_value))
        raise
    def tick(self, count = 1):
        self._curr_ticks += count
        if self._curr_ticks >= self._next_threshold:
            self._next_threshold += self._print_period
            percentage = 100 * self._curr_ticks // self._all_ticks
            percentage = wrap_color(4, '{:3d}%'.format(percentage))
            self._push_label(percentage)
    def _push_label(self, label):
        self._lineman.reprint(self._prefix + label)
    def info(self, message):
        self._parent.info(message)
    def warn(self, message):
        self._parent.warn(message)
    def fail(self, message):
        self._parent.fail(message)

class DevnullLog:
    __dummy__ = True
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    def subtask(self, *args, **kwargs):
        return self
    def progress(self, *args, **kwargs):
        return self
    def tick(self, *args, **kwargs):
        pass
    def info(self, message):
        pass
    def warn(self, message):
        pass
    def fail(self, message):
        pass

class ConsoleLog:
    __dummy__ = False
    def __init__(self, file = stdout, **kwargs):
        self._use_timer = kwargs.get('timer', False)
        line_width = 80
        if file.isatty():
            line_width = get_terminal_size().columns
        self._lineman = LineManager(line_width, file)
    def subtask(self, description_line, **kwargs):
        return SingleSubtask(self, description_line, **kwargs)
    def progress(self, description_line, ticks):
        return SpinnerSubtask(self, description_line, ticks)
    def info(self, message):
        self._lineman.push(wrap_color(6, '[INFO] '), None, 7)
        self._lineman.message(message)
        self._lineman.pop()
    def warn(self, message):
        self._lineman.push(wrap_color(7, '[WARN] '), None, 7)
        self._lineman.message(message)
        self._lineman.pop()
    def fail(self, message):
        self._lineman.push(wrap_color(3, '[FAIL] '), None, 7)
        self._lineman.message(message)
        self._lineman.pop()
