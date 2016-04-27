#!/usr/bin/python3
from CVE.Annotation.Dataset import DollarAnnotation, SimpleDetectionsList
from CVE import evaluate
from CVE.Role import asGtDataset, asEvalDataset
from CVE.Verifier import BoundingBoxIoUVerifier
from CVE.Driver import DetectionCurveDriver
from CVE.Plugin import Plotter2D, TuneBBoxesAnnotations
from CVE.Logger import ConsoleLog, set_default_logger

set_default_logger(ConsoleLog())

dataset1 = DollarAnnotation('/home/sergeev/bigsample-workload/gt')
dataset2 = SimpleDetectionsList('/home/sergeev/bigsample-workload/detections-min0.05.txt')

res = evaluate(
    asGtDataset(dataset1),
    asEvalDataset(dataset2),
    BoundingBoxIoUVerifier(threshold = 0.5),
    DetectionCurveDriver(mode = ('precision', 'recall')),
    Plotter2D('output-10.png', lang = 'en', label = 'facesdk, 1 round'),
    TuneBBoxesAnnotations(threshold = 0.95, rounds = 1),
)

print(res)
