#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:08:39 2021

@author: Neil J Thomson
"""

import os
import sys
#path_to_pensa_folder='pensa'
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(path_to_pensa_folder), '..')))
from pensa import *


"""
for the pdb visualisation, the trajectory needs to be fit to the first frame of the simulation
so that the density and protein align with each other
"""

# # # Get the water pocket data for the top 3 most probable sites (top_waters = 3).
# # # Orientation of the waters (spherical coordinates [radians]) is a timeseries distribution. 
# # # When water is not present at the site, the orientation is recorded as 10000.0 to represent an empty state.
# # # Visualise the pocket occupancies on the reference structure in a pdb file (write=True) with occupation frequencies
# # # saved as b_factors. If write=True, you must specify the water model for writing out the grid.
# # # options include:
# SPC	
# TIP3P
# TIP4P	
# water	
struc = "mor-data/11426_dyn_151.pdb"
xtc = "mor-data/11423_trj_151.xtc"
water_feat, water_data = get_water_features(structure_input = struc, 
                                            xtc_input = xtc,
                                            top_waters = 1,
                                            atomgroup = "OH2",
                                            write = True,
                                            write_grid_as="TIP3P",
                                            out_name = "11426_dyn_151")

# # # # # If we have already obtained the grid, we can speed up featurization by reading it in.
# # # # # Here we add the write=True option to write out the 
struc = "mor-data/11426_dyn_151.pdb"
xtc = "mor-data/11423_trj_151.xtc"
grid = "water_features/11426_dyn_151OH2_density.dx"
water_feat, water_data = get_water_features(structure_input = struc, 
                                            xtc_input = xtc,
                                            top_waters = 1,
                                            atomgroup = "OH2",
                                            grid_input = grid)




# # # We can use the get_atom_features, which provides the same
# # # functionality but ignores orientations as atoms are considered spherically symmetric.
struc = "mor-data/11426_dyn_151.pdb"
xtc = "mor-data/11423_trj_151.xtc"
# # # Here we locate the sodium site which has the highest probability
# # # The density grid is written (write=True) using the default density conversion "Angstrom^{-3}" in MDAnalysis
atom_feat, atom_data = get_atom_features(structure_input = struc, 
                                         xtc_input = xtc,
                                         top_atoms = 1,
                                         atomgroup = "SOD",
                                         element = "Na",
                                         write = True,
                                         out_name = "11426_dyn_151")

# # # If we have already written a grid, we can set that as input 
# # # using the argument grid_input="gridname.dx" to speed up the featurization
