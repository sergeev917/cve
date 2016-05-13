#!/usr/bin/python3
from CVE.Annotation.Dataset import DollarAnnotation, SimpleDetectionsList
from CVE import evaluate
from CVE.Role import asGtDataset, asEvalDataset
from CVE.Verifier import BoundingBoxIoUVerifier
from CVE.Driver import DetectionCurveDriver
from CVE.Plugin import Plotter2D, TuneBBoxesAnnotations
from CVE.Logger import ConsoleLog, set_default_logger

set_default_logger(ConsoleLog(timer = True))

dataset1 = DollarAnnotation('/home/sergeev/bigsample-workload/gt')
dataset2 = SimpleDetectionsList('/home/sergeev/bigsample-workload/detections-min0.05.txt')

#res = evaluate(
#    asGtDataset(dataset1),
#    asEvalDataset(dataset2),
#    BoundingBoxIoUVerifier(threshold = 0.5),
#    DetectionCurveDriver(mode = ('precision', 'recall')),
#    TuneBBoxesAnnotations(threshold = 0.95, rounds = 5),
#    Plotter2D('/tmp/output-10.png', lang = 'en', label = 'facesdk, 5 rounds'),
#)
verifier = BoundingBoxIoUVerifier()
verifier.reconfigure(dataset1, dataset2)

class ValueStorage:
    def __init__(self):
        self.value = None
    def make_getter(self):
        return lambda: self.value

def f1():
    x = 0
    for name, markup in dataset1:
        r = verifier(markup, dataset2[name])
        x += r[3]
    return x

def run_instructions(lst):
    storage = [None] * 3
    for func, args, out in lst:
        if len(args) == 0:
            storage[out] = func()
        else:
            storage[out] = func(*map(storage.__getitem__, args))
    return storage[-1]


def f2():
    x = 0
    markup1_storage = ValueStorage()
    samplename_storage = ValueStorage()
    instructions = [
        (markup1_storage.make_getter(), [], 0),
        (lambda: dataset2[samplename_storage.value], [], 1),
        (verifier, [0, 1], 2),
    ]
    for name, markup in dataset1:
        markup1_storage.value = markup
        samplename_storage.value = name
        r = run_instructions(instructions)
        x += r[3]
    return x

def f3():
    x = 0
    storage = [None, None]
    instructions = [
        (lambda: storage[0], [], 0),
        (lambda: dataset2[storage[1]], [], 1),
        (verifier, [0, 1], 2),
    ]
    for name, markup in dataset1:
        storage[0] = markup
        storage[1] = name
        r = run_instructions(instructions)
        x += r[3]
    return x

def f4():
    x = 0
    storage = [None, None]
    instructions = [
        (lambda: storage[1], [], 0),
        (lambda: dataset2[storage[0]], [], 1),
        (verifier, [0, 1], 2),
    ]
    for element in dataset1:
        storage[:] = element
        r = run_instructions(instructions)
        x += r[3]
    return x

import IPython
IPython.start_ipython(user_ns = locals())
print('f1()', IPython.get_ipython().magic('timeit -r100 f1()'))
print('f2()', IPython.get_ipython().magic('timeit -r100 f2()'))
print('f3()', IPython.get_ipython().magic('timeit -r100 f3()'))
print('f4()', IPython.get_ipython().magic('timeit -r100 f4()'))

#f1()
#f2()
