#####################################################################
#                                                                   #
# /labscript_devices/FunctionRunner/__init__.py                     #
#                                                                   #
# Copyright 2019, Monash University and contributors                #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################

from labscript_utils import check_version

import sys
if sys.version_info < (3, 5):
    raise RuntimeError("FunctionRunner requires Python 3.5+")