#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:54:24 2023

@author: neil
"""

import os
from pensa import *


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # Featurizing water cavity hydrogen bonds  

root_dir = './mor-data'

ref_file_a =  root_dir+'/11427_dyn_151.psf'
pdb_file_a =  root_dir+'/11426_dyn_151.pdb'
trj_file_a =  root_dir+'/11423_trj_151.xtc'

names, data = get_cavity_bonds(ref_file_a, trj_file_a,
                                atomgroups = ['OH2', 'H1', 'H2'], 
                                site_IDs = [1,2], 
                                grid_input=None,
                                write=True,
                                write_grid_as='TIP3P',
                                out_name='11423_trj_169')


# # # # Featurizing ligand-protein hydrogen bonds  
ref_file_a =  root_dir+'/11580_dyn_169.psf'
pdb_file_a =  root_dir+'/11579_dyn_169.pdb'
trj_file_a =  root_dir+'/11578_trj_169.xtc'

names, data = get_h_bonds(ref_file_a, trj_file_a,
                            fixed_group = 'resname 4VO',
                            dyn_group='protein', 
                            write=True,
                            out_name='4VO_hbonds')




