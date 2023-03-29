#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:03:52 2022

@author: neil
"""
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
  HydrogenBondAnalysis as HBA)
import numpy as np
import itertools as it
from gridData import Grid
from tqdm import tqdm
import os
from pensa.preprocessing.density import *

def Unique_bonding_pairs(lst):
     return ([list(i) for i in {*[tuple(sorted(i)) for i in lst]}])
 
def get_cavity_bonds(structure_input, xtc_input, atomgroups, site_IDs, 
                     grid_input=None, write=None, write_grid_as=None, out_name=None):
    """
    Find hydrogen bonds between waters occupying cavities and protein.
    
    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    site_IDs : list
        List of indexes for the sites desired to investigate.
    grid_input : str, optional
        File name for the density grid input. The default is None, and a grid is automatically generated.
    write : bool, optional
        If true, the following data will be written out: reference pdb with occupancies,
        water distributions, water data summary. The default is None.
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model 
        to convert the density into. The default is None. Options are suggested if default.
    out_name : str, optional
        Prefix for all written filenames. The default is None.
    
    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features
    
    """
    
    if write is not None:
        if out_name is None:
            print('WARNING: You are writing results without providing out_name.')
    
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
        
    u = mda.Universe(structure_input, xtc_input)
    if write is True:
        if not os.path.exists('h2o_hbonds/'):
            os.makedirs('h2o_hbonds/')
        p = u.select_atoms("protein")
        pdb_outname = 'h2o_hbonds/' + out_name + "_Sites.pdb"
        p_avg = np.zeros_like(p.positions)
        # do a quick average of the protein (in reality you probably want to remove PBC and RMSD-superpose)
        for ts in u.trajectory:
            p_avg += p.positions
        p_avg /= len(u.trajectory)
        # temporarily replace positions with the average
        p.positions = p_avg
        # write average protein coordinates
        p.write(pdb_outname)
        # just make sure that we have clean original coordinates again (start at the beginning)
        u.trajectory.rewind()        
        
        if grid_input is None:
            g = get_grid(u, atomgroups[0], write_grid_as,  out_name)           
        else:
            g = Grid(grid_input)  
    elif grid_input is None:
        g = get_grid(u, atomgroup)              
    else:
        g = Grid(grid_input)  
    
    xyz, val = local_maxima_3D(g.grid)
    ## Negate the array to get probabilities in descending order
    val_sort = np.argsort(-1*val.copy())
    coords = [xyz[max_val] for max_val in val_sort]    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    site_information=[]
    O_hbonds_all_site=[]
    H_hbonds_all_site=[]
    
    
    for site_no in site_IDs:
        
        feature_names['W'+str(site_no)]={}
        features_data['W'+str(site_no)]={}
        
        ID_to_idx = site_no - 1
        print('\n')
        print('Site no: ',site_no)
        print('\n')
        O_hbonds = []
        H_hbonds = []
        ## Find all water atoms within 3.5 Angstroms of density maxima
        # Shifting the coordinates of the maxima by the grid origin to match 
        # the simulation box coordinates
        shifted_coords=coords[ID_to_idx]+g.origin
        point_str = str(shifted_coords)[1:-1]
        counting=[]
        
        if write is True:
            write_atom_to_pdb(pdb_outname, shifted_coords, 'W'+str(site_no), atomgroups[0])
                
        for frame_no in tqdm(range(len(u.trajectory))):       
        # for frame_no in tqdm(range(100)):       
            u.trajectory[frame_no]
            radius = ' 3.5'
            atomgroup_IDS = list(u.select_atoms('byres (name ' + atomgroups[0] + ' and point ' + point_str + radius + ')').indices)[::3]
            counting.append(list(set(atomgroup_IDS)))
    
        ## Water atom indices that appear in the water site
        flat_list = [item for sublist in counting for item in sublist]
        ## Extract water orientation timeseries
        for frame_no in tqdm(range(len(u.trajectory))):   
        # for frame_no in tqdm(range(100)):       

            u.trajectory[frame_no]
            site_resid=counting[frame_no]
            # print(site_resid)
            if len(site_resid)==1:        
                ## (x,y,z) positions for the water oxygen at trajectory frame_no
                proxprotatg = 'protein and around 3.5 byres index ' + str(site_resid[0])
                O_site = 'index ' + str(site_resid[0]) 
                H_site = '((byres index ' +  str(site_resid[0]) + ') and (name '+atomgroups[1]+' or name '+atomgroups[2]+'))'
                
                hbond = HBA(universe=u)
                protein_hydrogens_sel = hbond.guess_hydrogens(proxprotatg)
                protein_acceptors_sel = hbond.guess_acceptors(proxprotatg)
                if len(protein_hydrogens_sel)!=0:
                # bonds formed by the water oxygen
                    O_bonds = '( '+protein_hydrogens_sel+' ) and around 3.5 '+O_site
                else:
                    O_bonds = ''
                # bonds formed by the water hydrogens
                if len(protein_acceptors_sel)!=0:
                    H_bonds = '( '+ protein_acceptors_sel+' ) and around 3.5 '+H_site
                else:
                    H_bonds=''
    
                H_hbond_group = u.select_atoms(H_bonds)
                H_hbonds.append(H_hbond_group)
                O_hbond_group = u.select_atoms(O_bonds)
                O_hbonds.append(O_hbond_group)    
                
            ## Featurize water with highest pocket occupation (if multiple waters in pocket)
            elif len(site_resid)>1:
                freq_count=[]
                for ID in site_resid:
                    freq_count.append([flat_list.count(ID),ID])
                freq_count.sort(key = lambda x: x[0])
    
                proxprotatg = 'protein and around 3.5 byres index ' + str(freq_count[-1][1])
                O_site = 'index ' + str(freq_count[-1][1])
                H_site = '((byres index ' + str(freq_count[-1][1]) + ') and (name '+atomgroups[1]+' or name '+atomgroups[2]+'))'
                
                hbond = HBA(universe=u)
                protein_hydrogens_sel = hbond.guess_hydrogens(proxprotatg)
                
                protein_acceptors_sel = hbond.guess_acceptors(proxprotatg)
                if len(protein_hydrogens_sel)!=0:
                # bonds formed by the water oxygen
                    O_bonds = '( '+protein_hydrogens_sel+' ) and around 3.5 '+O_site
                else:
                    O_bonds = ''                
                # bonds formed by the water hydrogens
                if len(protein_acceptors_sel)!=0:
                    H_bonds = '( '+ protein_acceptors_sel+' ) and around 3.5 '+H_site
                else:
                    H_bonds=''

                H_hbond_group = u.select_atoms(H_bonds)
                H_hbonds.append(H_hbond_group)
                O_hbond_group = u.select_atoms(O_bonds)
                O_hbonds.append(O_hbond_group)    
            
            ## 10000.0 = no waters bound
            elif len(site_resid)<1:
                O_hbonds.append("unocc")
                H_hbonds.append("unocc")
        
        
        bondouts=[]
        for bondtype in [O_hbonds,H_hbonds]:
            resids = []
            for line in bondtype:
                if type(line) is str:
                    resids.append([line])
                else:
                    idxes = [8, 10, 2]
                    all_atgs=[]
                    for atg in range(len(line)):
                        # print(line[atg])
                        stringdex = [str(line[atg]).split(' ')[idx] for idx in idxes]
                        all_atgs.append(stringdex[0][:-1]+" "+stringdex[1]+" "+stringdex[2])
                    resids.append(all_atgs)
                    
            names = list(set([flat for sub in resids for flat in sub]))
            if names.count('unocc')>0:
                names.remove('unocc')
            dist = np.zeros((len(names),len(u.trajectory)))     
            for bondsite in range(len(names)):
                for frame in range(len(resids)):
                    if resids[frame].count(names[bondsite])>0:
                        dist[bondsite][frame]=1
            bondouts.append([names,dist])
            
        O_site_pdb_id = "O" + str(site_no)
        H_site_pdb_id = "H" + str(site_no)
        # Write data out and visualize water sites in pdb         
        "FIX OUTPUT UNIFORMITY, SINGLE BONDS NOT OUTPUT WITH ANY ARRAY DIMENSION"
        if write is True:    
            np.savetxt('h2o_hbonds/' + out_name + O_site_pdb_id + '_names.txt', np.array(bondouts[0][0],dtype=object), fmt='%s')
            np.savetxt('h2o_hbonds/' + out_name + O_site_pdb_id + '_data.txt',  np.array(bondouts[0][1],dtype=object), fmt='%s')
            np.savetxt('h2o_hbonds/' + out_name + H_site_pdb_id + '_names.txt', np.array(bondouts[1][0],dtype=object), fmt='%s')
            np.savetxt('h2o_hbonds/' + out_name + H_site_pdb_id + '_data.txt',  np.array(bondouts[1][1],dtype=object), fmt='%s')
    
    
        feature_names['W'+str(site_no)]['acceptor_names'] = np.array(bondouts[0][0],dtype=object)
        feature_names['W'+str(site_no)]['donor_names'] = np.array(bondouts[1][0],dtype=object)
        features_data['W'+str(site_no)]['acceptor_timeseries'] = np.array(bondouts[0][1],dtype=object)
        features_data['W'+str(site_no)]['donor_timeseries'] = np.array(bondouts[1][1],dtype=object)
        features_data['W'+str(site_no)]['acceptor_frequencies'] = np.sum(np.array(bondouts[0][1],dtype=object),axis=1)/len(u.trajectory)
        features_data['W'+str(site_no)]['donor_frequencies'] = np.sum(np.array(bondouts[1][1],dtype=object),axis=1)/len(u.trajectory)
        
    return feature_names, features_data
 
def get_h_bonds(structure_input, xtc_input, atomgroup1, atomgroup2, write=None, out_name=None):
    """
    Find hydrogen bonding partners for atomgroup1 in atomgroup2.
    
    Parameters
    ----------
    structure_input : str
        File name for the reference file (TPR format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    UPDATE ARGUMENTS

    write : bool, optional
        If true, the following data will be written out: reference pdb with occupancies,
        water distributions, water data summary. The default is None.
    out_name : str, optional
        Prefix for all written filenames. The default is None.
    
    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features
    
    """
    
    if write is not None:
        if out_name is None:
            print('WARNING: You are writing results without providing out_name.')
    
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
        
    u = mda.Universe(structure_input, xtc_input)
    if write is True:
        if not os.path.exists('lig_hbonds/'):
            os.makedirs('lig_hbonds/')
            
    # First locate all potential bonding sites 
    interacting_atoms1 = atomgroup1
    interacting_atoms2 = atomgroup2 + " and around 3.5 " + atomgroup1
    
    ag1_atoms = list(u.select_atoms(atomgroup1).indices)
    
    # bonds for [[atomgroup1 donors] ,[atomgroup1 acceptors]]
    all_bonds = [[],[]]    
    for frame_no in tqdm(range(len(u.trajectory))):   
    # for frame_no in tqdm(range(100)):       
        u.trajectory[frame_no]
        # First locate all potential bonding sites for atomgroups
        hbond = HBA(universe=u)
        atomgroup_donors1 = hbond.guess_hydrogens(interacting_atoms1)
        atomgroup_acceptors1 = hbond.guess_acceptors(interacting_atoms1)    

        atomgroup_donors2 = hbond.guess_hydrogens(interacting_atoms2)
        atomgroup_acceptors2 = hbond.guess_acceptors(interacting_atoms2)   
        # print(atomgroup_donors1,atomgroup_acceptors1,atomgroup_donors2,atomgroup_acceptors2)
        # sorting the atomgroups into pairs for only inter bonds between ag1 and ag2 and not intra
        bond_atomgroup_pairs = [[atomgroup_donors1,atomgroup_acceptors2],[atomgroup_acceptors1,atomgroup_donors2]]
        
        append_counter = 0
        for b_atg_p in bond_atomgroup_pairs:
            # obtain indices for all donor and acceptor atoms
            donor_idcs = u.select_atoms(b_atg_p[0]).indices
            acceptor_idcs = u.select_atoms(b_atg_p[1]).indices

            frame_bonds = []
            for donor_idx in donor_idcs:
                idx_bonds = []
                # find donor positions
                donor_pos = np.array(u.select_atoms("index "+str(donor_idx)).positions)
                for acceptor_idx in acceptor_idcs:
                    # find acceptor positions
                    acceptor_pos = np.array(u.select_atoms("index "+str(acceptor_idx)).positions)
                    # if distance between atoms less than 3.5 angstrom then count as bond
                    if np.linalg.norm(donor_pos-acceptor_pos) < 3.5:
                        idx_bonds.append([donor_idx,acceptor_idx])
                frame_bonds.append(idx_bonds)
            all_bonds[append_counter].append(frame_bonds)
            append_counter+=1
    
    all_donor_pairs = Unique_bonding_pairs([y for subl in [x for sub in all_bonds[0] for x in sub] for y in subl])
    all_acceptor_pairs = Unique_bonding_pairs([y for subl in [x for sub in all_bonds[1] for x in sub] for y in subl])
    
    all_donor_pair_names = [[atg_to_names(u.select_atoms('index '+str(i[0])))[0],atg_to_names(u.select_atoms('index '+str(i[1])))[0]] for i in all_donor_pairs]
    all_acceptor_pair_names = [[atg_to_names(u.select_atoms('index '+str(i[0])))[0],atg_to_names(u.select_atoms('index '+str(i[1])))[0]] for i in all_acceptor_pairs]
    
    donor_dist =np.zeros((len(all_donor_pairs),len(u.trajectory)))
    acceptor_dist = np.zeros((len(all_acceptor_pairs),len(u.trajectory)))

    for frame in tqdm(range(len(u.trajectory))):
        for pair in range(len(all_donor_pairs)):
            if list(reversed(all_donor_pairs[pair])) in [flat for sub in all_bonds[0][frame] for flat in sub]:
                donor_dist[pair][frame]=1
        for pair in range(len(all_acceptor_pairs)):
            if list(reversed(all_acceptor_pairs[pair])) in [flat for sub in all_bonds[1][frame] for flat in sub]:
                acceptor_dist[pair][frame]=1


    # Write data out and visualize water sites in pdb           
    if write is True:    
        np.savetxt('lig_hbonds/'+out_name+'all_donor_pair_names.txt', np.array(all_donor_pair_names,dtype=object), fmt='%s')
        np.savetxt('lig_hbonds/'+out_name+'all_acceptor_pair_names.txt',  np.array(all_acceptor_pair_names,dtype=object), fmt='%s')
        np.savetxt('lig_hbonds/'+out_name+'all_donor_pair_data.txt', np.array(donor_dist,dtype=object), fmt='%s')
        np.savetxt('lig_hbonds/'+out_name+'all_acceptor_pair_data.txt',  np.array(acceptor_dist,dtype=object), fmt='%s')
    
    feature_names['donor_names'] = np.array(all_donor_pair_names)
    feature_names['acceptor_names'] = np.array(all_acceptor_pair_names)
    features_data['donor_data'] = np.array(donor_dist)
    features_data['accptor_data'] = np.array(acceptor_dist)
             
    return feature_names, features_data
 
def atg_to_names(atg):
    idxes = [8, 10, 2]
    all_atgs=[]
    print(atg)
    for line in range(len(atg)):
        stringdex = [str(atg[line]).split(' ')[idx] for idx in idxes]
        all_atgs.append(stringdex[0][:-1]+" "+stringdex[1]+" "+stringdex[2])        
    return all_atgs
 
    