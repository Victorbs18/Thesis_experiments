# src/algorithms.py
from domainbed.algorithms import (
    ERM, IRM, GroupDRO, CORAL, DANN, VREx)

ALGORITHMS = {
    'ERM':      ERM,
    'IRM':      IRM,
    'GroupDRO': GroupDRO,
    'CORAL':    CORAL,
    'DANN':     DANN,
    'VREx':     VREx,
}