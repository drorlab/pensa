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


"""
for the pdb visualisation, the trajectory needs to be fit to the first frame of the simulation
so that the density and protein align with each other
"""


struc = "mor-data/11426_dyn_151.gro"
xtc = "mor-data/11423_trj_151.xtc"
# # # Get the water site occupation frequencies of the top 10 most probable sites (top_waters = 10)
# # # And visualise them on the reference structure in a pdb file (pdb_vis=True) with occupation frequencies
# # # saved as b_factors.
water_frequencies = get_water_features(structure_input = struc, 
                                      xtc_input = xtc,
                                      atomgroup = "OH2",
                                      pdb_vis = True,
                                      top_waters = 10)

# # # The variable "water_frequencies" then contains:
# 1. A unique identifier for each water
# 2. An array with the (x,y,z) coordinates of the water site
# 3. The occupation frequency of that site.

# # # If we are using pdb_vis=True, the occupation frequency can be visualised in the 
# # # output pdb by selecting the b_factor_putty preset.

# # # If we want to write out the density grid to visualise with the structure
# # # Add the grid_wat_model option compatible with MDAnalysis density.convert

# # # options include:
# SPC	
# TIP3P
# TIP4P	
# water	
water_frequencies = get_water_features(structure_input = struc, 
                                      xtc_input = xtc,
                                      atomgroup = "OH2",
                                      grid_wat_model="TIP3P",
                                      top_waters = 5)

# # # # If we have already obtained the grid, we can speed up featurization by reading it in.
# # # # Here we add the write=True option to write out the 
# # # # orientation of the waters (spherical coordinates [radians]) as a timeseries distribution. 
# # # # When water is not present at the site, the orientation is recorded as 10000.0 to represent an empty state.
grid = "mor-data/11426_dyn_151OH2_density.dx"
water_frequencies = get_water_features(structure_input = struc, 
                                      xtc_input = xtc,
                                      atomgroup = "OH2",
                                      grid_input = grid,
                                      write = True,
                                      top_waters = 1)




# # # Say we are looking at ion channels, and we want to locate any stable/metastable
# # # ion binding sites, we can use the get_atom_features, which provides the same
# # # functionality but ignores orientations as atoms are considered spherically symmetric.
struc = "mor-data/11426_dyn_151.gro"
xtc = "mor-data/11423_trj_151.xtc"
# # # Here we locate the sodium site which has the highest frequency of occupation 
# # # The density grid is written (grid_write=True) using the default density conversion "Angstrom^{-3}" in MDAnalysis
# # # "atom_frequencies" contains the same information as "water_frequencies":
# 1. A unique identifier for each atom
# 2. An array with the (x,y,z) coordinates of the water site
# 3. The occupation frequency of that site.
# # # We decide not to write the atom_frequencies out on this occasion by setting write=None (default)
atom_frequencies = get_atom_features(structure_input = struc, 
                                    xtc_input = xtc,
                                    atomgroup = "SOD",
                                    element = "Na",
                                    top_atoms = 1,
                                    grid_write=True,
                                    write=None)
# # # If we have already written a grid, we can set that as input 
# # # using the argument grid_input="gridname.dx" to speed up the featurization

















