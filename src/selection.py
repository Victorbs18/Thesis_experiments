# src/selection.py
from domainbed.model_selection import (
    IIDAccuracySelectionMethod,
    LeaveOneOutSelectionMethod,
    OracleSelectionMethod,
)
from domainbed.lib.query import Q

SELECTION_METHODS = {
    'IIDAccuracySelectionMethod':  IIDAccuracySelectionMethod,
    'LeaveOneOutSelectionMethod':  LeaveOneOutSelectionMethod,
    'OracleSelectionMethod':       OracleSelectionMethod,
}