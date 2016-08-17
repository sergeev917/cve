#!/usr/bin/python3
from CVE.Annotation.Dataset import DollarAnnotation, SimpleDetectionsList
from CVE.Role import asGtDataset, asEvalDataset
from CVE.Verifier import BoundingBoxIoUVerifier
from CVE.Analysis import DetectionPerformanceCurve
from CVE.Logger import ConsoleLog, set_default_logger
from CVE.DatasetWalker import SimpleWalker
from CVE.Plugin import CurvePlotter2D
from CVE import evaluate

set_default_logger(ConsoleLog(timer = True))

print(evaluate(
    asGtDataset(DollarAnnotation('/home/sergeev/bigsample-workload/gt')),
    asEvalDataset(SimpleDetectionsList('/home/sergeev/bigsample-workload/detections-min0.05.txt')),
    BoundingBoxIoUVerifier(0.5),
    SimpleWalker(workers = None),
    DetectionPerformanceCurve(mode = ('precision', 'recall')),
    CurvePlotter2D('/tmp/imgout.png', label = 'BigSample test', lang = 'ru'),
))
