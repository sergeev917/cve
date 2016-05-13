#!/usr/bin/python3
from CVE.Annotation.Dataset import DollarAnnotation, SimpleDetectionsList
from CVE import evaluate
from CVE.Role import asGtDataset, asEvalDataset
from CVE.Verifier import BoundingBoxIoUVerifier
from CVE.Driver import DetectionCurveDriver
from CVE.Plugin import Plotter2D, TuneBBoxesAnnotations
from CVE.Logger import ConsoleLog, set_default_logger
from CVE.FlowGraph import DependencyFlowManager

set_default_logger(ConsoleLog(timer = True))

class DependencyFlowNode:
    def __init__(self, contracts):
        self._contracts = []
        for mode_id, data in enumerate(contracts):
            self._contracts.append((mode_id, data[0], data[1]))
    def static_recipes(self):
        return self._contracts
man = DependencyFlowManager()
node_a = DependencyFlowNode([([], ['a'])])
man.add_node(node_a)
node_aa = DependencyFlowNode([([], ['a'])])
man.add_node(node_aa)
node_b = DependencyFlowNode([(['a'], ['a', 'b'])])
man.add_node(node_b)

print(man._resources)
print(man._nodes)
