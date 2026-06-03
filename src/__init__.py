import sys
import os

# DomainBed submodule path 
_DOMAINBED_PATH = os.path.join(os.path.dirname(__file__), '..', 'DomainBed')
sys.path.insert(0, os.path.abspath(_DOMAINBED_PATH))