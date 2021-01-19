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
# out_name_a = "traj/condition-a"
# out_name_b = "traj/condition-b"
# out_name_combined="traj/combined"

# for subdir in ['traj','plots','vispdb','pca','clusters','results']:
#     if not os.path.exists(subdir):
#         os.makedirs(subdir)

# # # # Extract the coordinates of the receptor from the trajectory
# extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a)
# extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b)
     
# start_frame=0  
# a_rec = get_features(out_name_a+"_receptor.gro",
#                       out_name_a+"_receptor.xtc", 
#                       start_frame)

# a_rec_feat, a_rec_data = a_rec

# b_rec = get_features(out_name_b+"_receptor.gro",
#                       out_name_b+"_receptor.xtc", 
#                       start_frame)

# b_rec_feat, b_rec_data = b_rec

# for k in a_rec_data.keys(): 
#     print(k, a_rec_data[k].shape)


# for k in b_rec_data.keys(): 
#     print(k, b_rec_data[k].shape)

out_name_a = "condition-a"
out_name_b = "condition-b"
# # # # Extract the multivariate torsion coordinates of each residue as a timeseries from the trajectory
# multivar_res_timeseries_data_a=multivar_res_timeseries_data(a_rec_feat,a_rec_data,'sc-torsions',write=True,out_name=out_name_a)
# multivar_res_timeseries_data_b=multivar_res_timeseries_data(b_rec_feat,b_rec_data,'sc-torsions',write=True,out_name=out_name_b)


# # # Parse data for SSI 
folder='sc-torsions/'
names=get_filenames(folder)
residues = list(set([item[len('condition-a'):] for item in names]))
res_sequence_order=[]
for r in residues: 
    j=re.split('(\d+)',r)
    res_sequence_order.append(j)
res_sequence_order.sort(key = lambda x: int(x[1]))
filename_sequence_ordered=[i[0]+i[1]+i[2] for i in res_sequence_order]

ssi=[]

count_empty=[]
for i in filename_sequence_ordered:
    
    if len(list(import_distribution(folder,out_name_a+i))) == 0: 
        count_empty.append(1)
        print('empty distribution')
    elif len(list(import_distribution(folder,out_name_a+i))) > 10: 
        dist_a=[list(import_distribution(folder,out_name_a+i))]
        dist_b=[list(import_distribution(folder,out_name_b+i))]    
    else:    
        dist_a=[list(i) for i in import_distribution(folder,out_name_a+i)]
        dist_b=[list(i) for i in import_distribution(folder,out_name_b+i)]
        
    if len(list(import_distribution(folder,out_name_a+i))) != 0: 
        
        combined_dist=[]
        for j in range(len(dist_a)):
            sim1,sim2=match_sim_lengths(dist_a[j],dist_b[j])
            combined_dist.append(sim1+sim2)
            # determine_state_limits(periodic_correction(combined_dist[-1]),show_plots=True)
            
        ssi.append(calculate_ssi(combined_dist,show_plots=True))
    
        print(i,ssi[-1])
        
        
np.savetxt('sc_ssi.txt',np.array(ssi))
