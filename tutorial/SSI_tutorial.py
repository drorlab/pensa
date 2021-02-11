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

for subdir in ['traj','plots','vispdb','pca','clusters','results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# # # # Extract the coordinates of the receptor from the trajectory
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


# # # # # Extract the multivariate torsion coordinates of each residue as a 
# # # # # timeseries from the trajectory and write into subdirectory   
# # # # # output = [[torsion 1 timeseries],[torsion 2 timeseries],...,[torsion n timeseries]]
multivar_res_timeseries_data_a=multivar_res_timeseries_data(a_rec_feat,a_rec_data,'sc-torsions',write=True,out_name=out_name_a)
multivar_res_timeseries_data_b=multivar_res_timeseries_data(b_rec_feat,b_rec_data,'sc-torsions',write=True,out_name=out_name_b)


# # # Parsing data for SSI 
folder='sc-torsions/'
out_name_a = "condition-a"
out_name_b = "condition-b"
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
# # # Calculate SSI one residue at a time on a read in basis
for residue in filename_sequence_ordered:
    
    residue_name = residue[:-4]
    # # # if torsion phase space is only 1 dimensional
    if import_distribution(folder,out_name_a+residue).ndim ==1: 
        dist_a=[list(import_distribution(folder,out_name_a+residue))]
        dist_b=[list(import_distribution(folder,out_name_b+residue))]    
    # # # if torsion phase space is multi-dimensional
    else:    
        dist_a=[list(i) for i in import_distribution(folder,out_name_a+residue)]
        dist_b=[list(i) for i in import_distribution(folder,out_name_b+residue)]
    
    combined_dist=[]
    for dist_no in range(len(dist_a)):
        # # # Make sure the ensembles have the same length of trajectory
        sim1,sim2=match_sim_lengths(dist_a[dist_no],dist_b[dist_no])
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

# # If we want to calculate the SSI shared between two residues, 
# # for example Arg165(R3.50) and Asn332(N7.49), we can input both distributions
# # into calculate_ssi()
folder='sc-torsions/'
out_name_a = "condition-a"
out_name_b = "condition-b"
    
Tyr336_dist_a=[list(i) for i in import_distribution(folder,out_name_a+'TYR336.txt')]
Tyr336_dist_b=[list(i) for i in import_distribution(folder,out_name_b+'TYR336.txt')]
Tyr336_combined_dist=[]
for j in range(len(Tyr336_dist_a)):
    # # # Make sure the ensembles have the same length of trajectory
    sim1,sim2=match_sim_lengths(Tyr336_dist_a[j],Tyr336_dist_b[j])
    # # # combine the ensembles into one distribution (condition_a + condition_b)
    Tyr336_combined_dist.append(sim1+sim2)
    
Asn332_dist_a=[list(i) for i in import_distribution(folder,out_name_a+'ASN332.txt')]
Asn332_dist_b=[list(i) for i in import_distribution(folder,out_name_b+'ASN332.txt')]  
Asn332_combined_dist=[]
for j in range(len(Asn332_dist_a)):
    # # # Make sure the ensembles have the same length of trajectory
    sim1,sim2=match_sim_lengths(Asn332_dist_a[j],Asn332_dist_b[j])
    # # # combine the ensembles into one distribution (condition_a + condition_b)
    Asn332_combined_dist.append(sim1+sim2)

# # # SSI between Arg165 and Asn332 is then calculated by running
ssi = calculate_ssi(Tyr336_combined_dist, Asn332_combined_dist, gauss_bins=10, write_plots=True,write_name="TEST")

# # # Lets say we want to calculate what magnitude of information
# # # shared between Tyr336 and Asn332 is information about the switch between ensembles
# # # then we can use co-SSI. The output of cossi also includes SSI between the 
# # # two variables. 
ssi, cossi = calculate_cossi(Tyr336_combined_dist,Asn332_combined_dist)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

# # Before we extract the water densities, we need to first align the trajectories 
# # so that we can featurize water sites in both ensembles using the same coordinates
condition_a = mda.Universe(out_name_a+".gro",out_name_a+".xtc")
condition_b = mda.Universe(out_name_b+".gro",out_name_b+".xtc")

align.AlignTraj(condition_b,  # trajectory to align
                condition_a,  # reference
                select='name CA',  # selection of atoms to align
                filename= out_name_b + 'aligned.xtc',  # file to write the trajectory to
                match_atoms=True,  # whether to match atoms based on mass
                ).run()

# # # re-define the condition_b universe with the aligned trajectory 
condition_b_aligned = mda.Universe(out_name_b+".gro", out_name_b+"aligned.xtc")

# # # We then combine the ensembles into one universe
Combined_conditions = mda.Merge(condition_a.atoms, condition_b_aligned.atoms)

# # # Extract the coordinates from the trajectories to add to the new universe
aligned_coords_b = AnalysisFromFunction(copy_coords,
                                      condition_b_aligned.atoms).run().results

aligned_coords_a = AnalysisFromFunction(copy_coords,
                                        condition_a.atoms).run().results

# # # The density needs to be formed from an even contribution of both conditions
# # # otherwise it will be unevely biased towards one condition.
# # # So we match the simulation lengths first
sim1_coords, sim2_coords = match_sim_lengths(aligned_coords_a,aligned_coords_b)
# # # Then we merge the coordinates into one system
merged_coords = np.hstack([sim1_coords,
                            sim2_coords])
# # # We load in the merged coordinated into our new universe that contains
# # # the receptor in both conditions
Combined_conditions.load_new(merged_coords, format=MemoryReader)

# # # We extract the density grid from the combined condition universe
density_atomgroup = Combined_conditions.select_atoms("name OH2")
# a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
D = DensityAnalysis(density_atomgroup, delta=1.0)
D.run()
D.density.convert_density("TIP3P")
D.density.export("combined_OH2_density.dx", type="double")
                  
grid_combined = "combined_OH2_density.dx"

# # # Then we can extract the featurize the waters common to both simulations
water_frequencies_a = get_water_features(structure_input = out_name_a+".gro", 
                                          xtc_input = out_name_a+".xtc",
                                          atomgroup = "OH2",
                                          grid_input = grid_combined,
                                          write = True,
                                          top_waters = 10)

water_frequencies_b = get_water_features(structure_input = out_name_b+".gro", 
                                          xtc_input = out_name_b+"aligned.xtc",
                                          atomgroup = "OH2",
                                          grid_input = grid_combined,
                                          write = True,
                                          top_waters = 10)

# # # Calculating SSI is then exactly the same as for residues
folder='water_features/'
out_name_a = "cond-a_water"
out_name_b = "cond-b_water"
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





