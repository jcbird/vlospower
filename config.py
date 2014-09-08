"""
Name: config.py

Arguments: 

Keywords: 

Purpose:

"""
# Jonathan Bird
# Created:       2013-08-03
# Last modified: 2013-08-03
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Modules:
from __future__ import print_function, division, absolute_import
import os
import sys
#import time
import numpy as np
#import matplotlib.pyplot as plt
#import math as m
#import scipy as sp
#import itertools

##--------------------------------------------------------------------------##

_baseDir = os.getenv('BASE_DIR')
_rootDir = os.getenv('ROOT_DIR')
_rootFile = os.getenv('ROOT_FILE')

if not _baseDir:
    #environment variables were not assigned
    print ('Environment Variable $BASE_DIR not found, please load...')
    sys.exit()

pars = ['simdir_root', 'file_root', 'rotsim_dir', 'dc_dir', 'iord_dir', 'fig_dir']
vals = [_rootDir, _rootFile, _baseDir+'rotated_simsnaps/', _baseDir+'dc/', _baseDir+'iord/', _baseDir+'plots/']

params = {p : v for p,v in zip(pars,vals)}

