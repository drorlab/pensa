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
# root_dir = './mor-data'
# # Define which files to download
# md_files = ['11427_dyn_151.psf','11426_dyn_151.pdb', # MOR-apo
#             '11423_trj_151.xtc','11424_trj_151.xtc','11425_trj_151.xtc',
#             '11580_dyn_169.psf','11579_dyn_169.pdb', # MOR-BU72
#             '11576_trj_169.xtc','11577_trj_169.xtc','11578_trj_169.xtc']
# # Download all the files that do not exist yet
# for file in md_files:
#     if not os.path.exists(os.path.join(root_dir,file)):
#         download_from_gpcrmd(file,root_dir)

# root_dir = './mor-data'
# # Simulation A
# ref_file_a =  root_dir+'/11427_dyn_151.psf'
# pdb_file_a =  root_dir+'/11426_dyn_151.pdb'
# trj_file_a = [root_dir+'/11423_trj_151.xtc',
#               root_dir+'/11424_trj_151.xtc',
#               root_dir+'/11425_trj_151.xtc']
# # Simulation B
# ref_file_b =  root_dir+'/11580_dyn_169.psf'
# pdb_file_b =  root_dir+'/11579_dyn_169.pdb'
# trj_file_b = [root_dir+'/11576_trj_169.xtc',
#               root_dir+'/11577_trj_169.xtc',
#               root_dir+'/11578_trj_169.xtc']
# # Base for the selection string for each simulation
# sel_base_a = "(not name H*) and protein"
# sel_base_b = "(not name H*) and protein"
# # Names of the output files
out_name_a = "traj/condition-a"
out_name_b = "traj/condition-b"

# # for subdir in ['traj','plots','vispdb','pca','clusters','results']:
# #     if not os.path.exists(subdir):
# #         os.makedirs(subdir)

# # # # # # Extract the coordinates of the receptor from the trajectory
# # extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
# # extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)
     
# # # Extract the features from the beginning (start_frame) of the trajectory
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
data_names, data_ssi = ssi_analysis(sc_multivar_res_feat_a['sc-torsions'],
                                    sc_multivar_res_data_a['sc-torsions'],
                                    sc_multivar_res_feat_b['sc-torsions'],
                                    sc_multivar_res_data_b['sc-torsions'], 
                                    write_plots=None,
                                    verbose=True)


# # # If we want to calculate the SSI shared between two residues, 
# # # for example Arg165(R3.50) and Asn332(N7.49), we can input both distributions
# # # into calculate_ssi()
folder='sc-torsions/'
out_name_a = "condition-a"
out_name_b = "condition-b"
    
Tyr336_dist_a=[list(i) for i in import_distribution(folder,out_name_a+'TYR336.txt')]
Tyr336_dist_b=[list(i) for i in import_distribution(folder,out_name_b+'TYR336.txt')]
Tyr336_combined_dist=[]
for j in range(len(Tyr336_dististribution (condition_a + condition_b):
     # # # Make sure the ensembles have the same length of trajectory
    sim1,sim2=match_sim_lengths(Tyr336_dist_a[j],Tyr336_dist_b[j])
    # # # combine the ensembles into one distribution (condition_a + condition_b)
    Tyr336_combined_dist.append(sim1+sim2)))
    
    
Asn332_dist_a=[list(i) for i in import_distribution(folder,out_name_a+'ASN332.txt')]
Asn332_dist_b=[list(i) for i in import_distribution(folder,out_name_b+'ASN332.txt')]  
Asn332_combined_dist=[]
for j in range(len(Asn332_dist_a)):
    # # # Make sure the ensembles have the same length of trajectory
    sim1,sim2=match_sim_lengths(Asn332_dist_a[j],Asn332_dist_b[j])
    # # # combine the ensembles into one distribution (condition_a + condition_b)
    Asn332_combined_dist.append(sim1+sim2)

# # # SSI between Asn332 and Tyr336 is then calculated by running
ssi = calculate_ssi(Tyr336_combined_dist, Asn332_combined_dist)

# # # Lets say we want to calculate what magnitude of information
# # # shared between Tyr336 and Asn332 is information about the switch between ensembles
# # # then we can use co-SSI. The output of cossi also includes SSI between the 
# # # two variables for comparison. 
ssi, cossi = calculate_cossi(Tyr336_combined_dist,Asn332_combined_dist)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
# # # Names of the output files
out_name_a = "traj/cond-a_water"
out_name_b = "traj/cond-b_water"

for subdir in ['traj','plots','vispdb','pca','clusters','results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# # # # Extract the coordinates of the receptor from the trajectory
extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a, sel_base_a)
extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b, sel_base_b)

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

# # # Then we  featurize the waters common to both simulations
water_names_a, water_data_a = get_water_features(structure_input = out_name_a+".gro", 
                                          xtc_input = out_name_a+"aligned.xtc",
                                          top_waters = 5,
                                          atomgroup = "OH2",
                                          grid_input = grid_combined,
                                          data_write = True,
                                          out_name = "cond_a")

water_names_b, water_data_b  = get_water_features(structure_input = out_name_b+".gro", 
                                          xtc_input = out_name_b+".xtc",
                                          top_waters = 5,
                                          atomgroup = "OH2",
                                          grid_input = grid_combined,
                                          data_write = True,
                                          out_name = "cond_b")

# # # Calculating SSI is then exactly the same as for residues
folder='water_features/'
out_name_a = "cond_a"
out_name_b = "cond_b"
O1_dist_a=[list(i) for i in import_distribution(folder,out_name_a+'O1.txt')]
O1_dist_b=[list(i) for i in import_distribution(folder,out_name_b+'O1.txt')]

O1_combined_dist=[]
for i in range(len(O1_dist_a)):
    
    sim1,sim2=match_sim_lengths(O1_dist_a[i], O1_dist_b[i])
    
    O1_combined_dist.append(sim1 + sim2)

# # # SSI shared between water O1 and the switch between ensemble conditions
ssi = calculate_ssi(O1_combined_dist,write_plots=True,write_name="water_O1")

# # # If we just want to investigate SSI shared between the water site occupancy
# # # We can add ignore the water orietation states by defining our own states
ssi = calculate_ssi(O1_combined_dist, a_states=[[min(O1_combined_dist[0]),min(O1_combined_dist[0])+np.pi*2,20000.0],
                                                [min(O1_combined_dist[1]),min(O1_combined_dist[1])+np.pi*2,20000.0]])





