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



# Define where to save the GPCRmd files
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
# Names of the output files
out_name_a = "traj/condition-a"
out_name_b = "traj/condition-b"
out_name_combined="traj/combined"

for subdir in ['traj','plots','vispdb','pca','clusters','results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# # # Extract the coordinates of the receptor from the trajectory
extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)
     
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
# # # # Extract the multivariate torsion coordinates of each residue as a 
# # # # timeseries from the trajectory and write into subdirectory   
# # # # output = [[torsion 1 timeseries],[torsion 2 timeseries],...,[torsion n timeseries]]
multivar_res_timeseries_data_a=multivar_res_timeseries_data(a_rec_feat,a_rec_data,'sc-torsions',write=True,out_name=out_name_a)
multivar_res_timeseries_data_b=multivar_res_timeseries_data(b_rec_feat,b_rec_data,'sc-torsions',write=True,out_name=out_name_b)


# # # Parsing data for SSI 
folder='sc-torsions/'
# # # Get names of all the torsion files within the folder
names=get_filenames(folder)
# # # Remove unique identifier for the different ensembles
residues = list(set([item[len(out_name_a):] for item in names]))
# # # Arrange the filenames in numberical order according to the protein sequence
res_sequence_order=[]
for r in residues: 
    j=re.split('(\d+)',r)
    res_sequence_order.append(j)
res_sequence_order.sort(key = lambda x: int(x[1]))
filename_sequence_ordered=[i[0]+i[1]+i[2] for i in res_sequence_order]


# # # Initialize list for SSI values
SSI_full_receptor=[]

count_empty=[]
# # # Calculate SSI one residue at a time on a read in basis
for i in filename_sequence_ordered:
    
    residue_name = i[:-4]
    
    # # # Notify that the residue has no distributions for this feature
    if len(list(import_distribution(folder,out_name_a+i))) == 0: 
        count_empty.append(1)
        print('empty distribution')
        
    # # # Generate nested list of distributions for multivariate phase space of that residue 
    elif len(list(import_distribution(folder,out_name_a+i))) > 10: 
        dist_a=[list(import_distribution(folder,out_name_a+i))]
        dist_b=[list(import_distribution(folder,out_name_b+i))]    
    # # # Generate nested list of distributions for multivariate phase space of that residue 
    else:    
        dist_a=[list(i) for i in import_distribution(folder,out_name_a+i)]
        dist_b=[list(i) for i in import_distribution(folder,out_name_b+i)]
    
    # # # Combine non-empty distributions from both ensembles (a and b) into one distribution
    if len(list(import_distribution(folder,out_name_a+i))) != 0: 
        
        combined_dist=[]
        for j in range(len(dist_a)):
            # # # Make sure the ensembles have the same length of trajectory
            sim1,sim2=match_sim_lengths(dist_a[j],dist_b[j])
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            combined_dist.append(sim1+sim2)
            
            
        # # # Calculate the SSI between a component and the binary switch between ensembles a and b       
        # # # Output = SSI (float)
        # # # We enable write_plots=True to inspect the state clustering of the residue sc-torsion distributions
        # # # Output plots are stored in .png format in "ssi_plots/WRITE_NAME_dist_number.png". 
        # # # Specify the residue name for write_name.
        ssi = calculate_ssi(combined_dist,write_plots=True,write_name=residue_name)   
        
        # # # If we notice that one of the residues is not clustering properly
        # # # we can adjust the clustering parameters - gauss_bins=120 (default), gauss_smooth=10 (default)
        ssi = calculate_ssi(combined_dist,write_plots=True,write_name=residue_name,gauss_bins=120,gauss_smooth=10)   
        # # # The write_plots option maintains a constant binning on the distribution of 360 bins (1 degree resolution)
        # # # So the data is not affected by altering the clustering binning or smoothing parameters.
    
        # # # add ssi values to list to be written
        SSI_full_receptor.append(ssi)    
        
np.savetxt('sc_ssi.txt',np.array(ssi))
