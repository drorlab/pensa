#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:08:39 2021

@author: neil
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("pensa"), '..')))
from pensa import *


struc = "mor-data/11426_dyn_151.gro"
xtc = "mor-data/11423_trj_151.xtc"
grid = "mor-data/11426_dyn_151OH2_density.dx"


# water_frequencies = get_water_features(structure_input = struc, 
#                                       xtc_input = xtc,
#                                       atomgroup = "OH2",
#                                       grid_input = grid,
#                                       # write_grid_as="TIP3P",
#                                       write = True,
#                                       top_waters = 10)

##describe argument options
atom_frequencies = get_atom_features(structure_input = struc, 
                                    xtc_input = xtc,
                                    atomgroup = "SOD",
                                    element = "Na",
                                    # grid_input = "OW_density.dx",
                                    top_atoms = 1,
                                    write=None)