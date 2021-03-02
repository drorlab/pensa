#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:08:39 2021

@author: Neil J Thomson
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("pensa"), '..')))
from pensa import *


# # Define where to save the GPCRmd files
root_dir = './mor-data'
# Define which files to download
md_files = ['11427_dyn_151.psf','11426_dyn_151.pdb', # MOR-apo
            '11423_trj_151.xtc','11424_trj_151.xtc','11425_trj_151.xtc',
            '11580_dyn_169.psf','11579_dyn_169.pdb', # MOR-BU72
            '11576_trj_169.xtc','11577_trj_169.xtc','11578_trj_169.xtc']
# Download all the files that do not exist yet
for file in md_files:
    if not os.path.exists(os.path.join(root_dir,file)):
        download_from_gpcrmd(file,root_dir)

root_dir = './mor-data'
# Simulation A
ref_file_a =  root_dir+'/11427_dyn_151.psf'
pdb_file_a =  root_dir+'/11426_dyn_151.pdb'
trj_file_a = [root_dir+'/11423_trj_151.xtc',
              root_dir+'/11424_trj_151.xtc',
              root_dir+'/11425_trj_151.xtc']
# Simulation B
ref_file_b =  root_dir+'/11580_dyn_169.psf'
pdb_file_b =  root_dir+'/11579_dyn_169.pdb'
trj_file_b = [root_dir+'/11576_trj_169.xtc',
              root_dir+'/11577_trj_169.xtc',
              root_dir+'/11578_trj_169.xtc']
# Base for the selection string for each simulation
sel_base_a = "(not name H*) and protein"
sel_base_b = "(not name H*) and protein"
# # Names of the output files
out_name_a = "traj/condition-a"
out_name_b = "traj/condition-b"

# # # for subdir in ['traj','plots','vispdb','pca','clusters','results']:
# # #     if not os.path.exists(subdir):
# # #         os.makedirs(subdir)

# # # # # # # Extract the coordinates of the receptor from the trajectory
# # # extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
# # # extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)
     
# # # # Extract the features from the beginning (start_frame) of the trajectory
start_frame=0  
a_rec = get_features(out_name_a+"_receptor.gro",
                      out_name_a+"_receptor.xtc", 
                      start_frame)

a_rec_feat, a_rec_data = a_rec

b_rec = get_features(out_name_b+"_receptor.gro",
                      out_name_b+"_receptor.xtc", 
                      start_frame)

b_rec_feat, b_rec_data = b_rec


out_name_a = "condition-a"
out_name_b = "condition-b"


# # # Extract the multivariate torsion coordinates of each residue as a 
# # # timeseries from the trajectory and write into subdirectory   
# # # output = [[torsion 1 timeseries],[torsion 2 timeseries],...,[torsion n timeseries]]
sc_multivar_res_feat_a, sc_multivar_res_data_a = multivar_res_timeseries_data(a_rec_feat,a_rec_data,'sc-torsions',write=True,out_name=out_name_a)
sc_multivar_res_feat_b, sc_multivar_res_data_b = multivar_res_timeseries_data(b_rec_feat,b_rec_data,'sc-torsions',write=True,out_name=out_name_b)

# # # We can calculate the State Specific Information (SSI) shared between the 
# # # ensemble switch and the combined ensemble residue conformations.
# # # Set write_plots=True to generate a folder with all the clustered states for each residue.
data_names, data_ssi = ssi_ensemble_analysis(sc_multivar_res_feat_a['sc-torsions'],sc_multivar_res_data_a['sc-torsions'],
                                              sc_multivar_res_feat_b['sc-torsions'],sc_multivar_res_data_b['sc-torsions'], 
                                              write_plots=None,
                                              verbose=True)

# # # We can calculate the State Specific Information (SSI) shared between the 
# # # ensemble switch and the combined ensemble residue conformations.
# # # Write_plots is disabled for this function as the combinatorial nature
# # # of comaring all features will generate the same plot features**2 times.
# # # To view clustered distributions, use ssi_ensemble_analysis, which is much quicker.
data_names, data_ssi = ssi_feature_analysis(sc_multivar_res_feat_a['sc-torsions'],sc_multivar_res_data_a['sc-torsions'],
                                            sc_multivar_res_feat_b['sc-torsions'],sc_multivar_res_data_b['sc-torsions'], 
                                            verbose=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Featurizing waters and calculating SSI 

# # # First we preprocess the trajectories to extract coordinates for protein 
# # # and waters.
root_dir = './mor-data'
# Simulation A
ref_file_a =  root_dir+'/11427_dyn_151.psf'
pdb_file_a =  root_dir+'/11426_dyn_151.pdb'
trj_file_a = [root_dir+'/11423_trj_151.xtc',
              root_dir+'/11424_trj_151.xtc',
              root_dir+'/11425_trj_151.xtc']
# Simulation B
ref_file_b =  root_dir+'/11580_dyn_169.psf'
pdb_file_b =  root_dir+'/11579_dyn_169.pdb'
trj_file_b = [root_dir+'/11576_trj_169.xtc',
              root_dir+'/11577_trj_169.xtc',
              root_dir+'/11578_trj_169.xtc']
# Base for the selection string for each simulation protein and all waters
sel_base_a = "protein or byres name OH2"
sel_base_b = "protein or byres name OH2"
# # # # Names of the output files
out_name_a = "traj/cond-a_water"
out_name_b = "traj/cond-b_water"

# for subdir in ['traj','plots','vispdb','pca','clusters','results']:
#     if not os.path.exists(subdir):
#         os.makedirs(subdir)

# # # # # Extract the coordinates of the receptor from the trajectory
# extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a, sel_base_a)
# extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b, sel_base_b)

# # # # Extract the coordinates of the ensemble a aligned to ensemble b 
extract_aligned_coords(out_name_a+".gro", out_name_a+".xtc", 
                        out_name_b+".gro", out_name_b+".xtc")


# # # # Extract the combined density of the waters in both ensembles a and b 
extract_combined_grid(out_name_a+".gro", out_name_a+"aligned.xtc", 
                      out_name_b+".gro", out_name_b+".xtc",
                      atomgroup="OH2",
                      write_grid_as="TIP3P",
                      out_name= "ab_grid_")

grid_combined = "ab_grid_OH2_density.dx"

# # # Then we featurize the waters common to both simulations
# # # We can do the same analysis for ions using the get_atom_features featurizer. 
water_feat_a, water_data_a = get_water_features(structure_input = out_name_a+".gro", 
                                                xtc_input = out_name_a+"aligned.xtc",
                                                top_waters = 2,
                                                atomgroup = "OH2",
                                                grid_input = grid_combined)
                                                # write = True,
                                                # out_name = "cond_a")

water_feat_b, water_data_b  = get_water_features(structure_input = out_name_b+".gro", 
                                                 xtc_input = out_name_b+".xtc",
                                                 top_waters = 2,
                                                 atomgroup = "OH2",
                                                 grid_input = grid_combined)
                                                  # write = True,
                                                  # out_name = "cond_b")


# # # Calculating SSI is then exactly the same as for residues

# # # SSI shared between waters and the switch between ensemble conditions
data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_Distr'],water_data_a['WaterPocket_Distr'],
                                             water_feat_b['WaterPocket_Distr'],water_data_b['WaterPocket_Distr'], 
                                             verbose=True)


# # # Alternatively we can see if the pocket occupancy (the presence/absence of water at the site) shares SSI
# # # Currently this is only enabled with ssi_ensemble_analysis. We need to turn off the periodic boundary conditions
# # # as the distributions are no longer periodic.
data_names, data_ssi = ssi_ensemble_analysis(water_feat_a['WaterPocket_OccupDistr'],water_data_a['WaterPocket_OccupDistr'],
                                             water_feat_b['WaterPocket_OccupDistr'],water_data_b['WaterPocket_OccupDistr'],
                                             wat_occupancy=True, pbc=False, verbose=True)

# # # In this example we can see that it is the presence or absence of water O1, 
# # # and not the orientation of the water site, that is more important in distinguishing between ensembles.
# # # Water 02 has no functional significance with respect to what these ensembles are investigating.

# # # If we want to find out the impact of water on the SSI between 
# # # features and the ensemble (ssi_ensemble_analysis), we can use co-SSI. 
# # # An equivalent interpretation of co-SSI is how much the switch between ensembles
# # # is involved in the communication between two features.
feat_names, cossi_feat_names, data_ssi, data_cossi = cossi_featens_analysis(sc_multivar_res_feat_a['sc-torsions'],sc_multivar_res_data_a['sc-torsions'],
                                                                            sc_multivar_res_feat_b['sc-torsions'],sc_multivar_res_data_b['sc-torsions'],  
                                                                            water_feat_a['WaterPocket_OccupDistr'],water_data_a['WaterPocket_OccupDistr'],
                                                                            water_feat_b['WaterPocket_OccupDistr'],water_data_b['WaterPocket_OccupDistr'],
                                                                            verbose=True)

# Sanity checks:
# SSI of a distribution cannot be lower than the occupancy of that distribution.
# Ensemble SSI must be between [0,1]
# CoSSI must be betwen [-1,1]
# CoSSI between the same component and the ensemble is equal to the ensemble SSI. 
# Feature SSI can be greater than 1, provided that the SSI <= the minimum entropy of either feature.

