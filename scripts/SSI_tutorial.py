#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:08:39 2021

@author: neil
"""

import os
from pensa import *
from pensa.statesinfo import *
import re


root_dir = "/home/neil/cluster/Tests/pensa/tutorial"
# # Simulation A
ref_file_a = root_dir+"/traj/testa.psf"
pdb_file_a = root_dir+"/traj/testa.pdb"
trj_file_a = root_dir+"/traj/testa.xtc"
# Simulation B
ref_file_b = root_dir+"/traj/testb.psf"
pdb_file_b = root_dir+"/traj/testb.pdb"
trj_file_b = root_dir+"/traj/testb.xtc"
# Base for the selection string for each simulation
sel_base_a = "protein"
sel_base_b = "protein"
# Names of the output files
out_name_a = "traj/testa"
out_name_b = "traj/testb"
# out_name_combined="traj/NK_E_protonate"

for subdir in ['traj','plots','vispdb','pca','clusters','results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        
# Load the selection and generate the strings
sel_string_a = load_selection("tutorial/selections/a_protein.txt", sel_base_a+" and ")
sel_string_b = load_selection("tutorial/selections/b_protein.txt", sel_base_b+" and ")
# # Extract the coordinates of the receptor from the trajectory
extract_coordinates(ref_file_a, pdb_file_a, [trj_file_a], out_name_a+"_receptor", sel_string_a)
extract_coordinates(ref_file_b, pdb_file_b, [trj_file_b], out_name_b+"_receptor", sel_string_b)       


     
start_frame=0  
a_rec = get_features(out_name_a+"_receptor.gro",
                     out_name_a+"_receptor.xtc", 
                     start_frame)

a_rec_feat, a_rec_data = a_rec

b_rec = get_features(out_name_b+"_receptor.gro",
                     out_name_b+"_receptor.xtc", 
                     start_frame)

b_rec_feat, b_rec_data = b_rec

for k in a_rec_data.keys(): 
    print(k, a_rec_data[k].shape)


for k in b_rec_data.keys(): 
    print(k, b_rec_data[k].shape)


multivar_res_timeseries_data_a=multivar_res_timeseries_data(a_rec_feat,a_rec_data,'sc-torsions',write=True,out_name='a_rec')
multivar_res_timeseries_data_b=multivar_res_timeseries_data(b_rec_feat,b_rec_data,'sc-torsions',write=True,out_name='b_rec')



folder='sc-torsions/'
names=get_filenames(folder)
residues = list(set([item[7:] for item in names]))

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
            determine_state_limits(periodic_correction(combined_dist[-1]),show_plots=True)
            
        ssi.append(calculate_ssi(combined_dist))
    
        print(i,ssi[-1])
        
        
np.savetxt('sc_ssi.txt',np.array(ssi))