# -*- coding: utf-8 -*-
"""
Methods to obtain a timeseries distribution for the atom/ion pockets' occupancies.

Atom pockets are defined as radius 2.5 Angstroms (based off of bond lengths between Na and O, and Ca and O) 
centered on the probability density maxima of the atoms.


The methods here are based on the following paper:

    |    Neil J. Thomson, Owen N. Vickery, Callum M. Ives, Ulrich Zachariae: 
    |    Ion-water coupling controls class A GPCR signal transduction pathways. 
    |    https://doi.org/10.1101/2020.08.28.271510
    
"""

import MDAnalysis as mda
import numpy as np
from gridData import Grid
from tqdm import tqdm
import os
from pensa.preprocessing.density import *



def get_atom_features(structure_input, xtc_input, atomgroup, element, top_atoms=10, 
                      grid_input=None, write=None, out_name=None):
    """
    Featurize atom pockets for the top X most probable atoms (top_atoms).
  
    
    Parameters
    ----------
    structure_input : str
        File name for the reference file (PDB or GRO format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    element : TYPE
        DESCRIPTION.
    top_atoms : int, optional
        Number of atoms to featurize. The default is 10.
    grid_input : str, optional
        File name for the density grid input. The default is None, and a grid is automatically generated.
    write : bool, optional
        If true, the following data will be written out: reference pdb with occupancies,
        atom distributions, atom data summary. The default is None.
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
            print('WARNING: You must provide out_name if writing out result.')
        
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}

    u = mda.Universe(structure_input, xtc_input)
    
    if write is True:
        if not os.path.exists('atom_features/'):
            os.makedirs('atom_features/')
        protein = u.select_atoms("protein")
        pdb_outname = 'atom_features/'+ out_name + element + "_Sites.pdb"
        u.trajectory[0]
        protein.write(pdb_outname)
        if grid_input is None:
            g = get_grid(u, atomgroup, "Angstrom^{-3}", out_name)        
        else:
            g = Grid(grid_input)  
    elif grid_input is None:
        g = get_grid(u, atomgroup)              
    else:
        g = Grid(grid_input)  
        
    ##converting the density to a probability
    atom_number = len(u.select_atoms('name ' + atomgroup))
    grid_data = np.array(g.grid)*atom_number/np.sum(np.array(g.grid))

    ##mask all probabilities below the average water probability
    average_probability_density = atom_number/sum(1 for i in grid_data.flat if i)
    ##mask all grid centers with density less than threshold density
    grid_data[grid_data <= average_probability_density] = 0.0
    
    xyz, val = local_maxima_3D(grid_data)
    ##negate the array to get descending order from most prob to least prob
    val_sort = np.argsort(-1*val.copy())
    # values = [val[i] for i in val_sort]    
    coords = [xyz[max_val] for max_val in val_sort]    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    
    atom_information=[]
    atom_dists=[]
    
    
    if top_atoms > len(coords):
        top_atoms = len(coords)  

    print('\n')
    print('Featurizing ',top_atoms,' Atoms')
    for atom_no in range(top_atoms):
        print('\n')
        print('Atom no: ',atom_no+1)
        print('\n')

        counting=[]
        ## Find all water atoms within 2.5 Angstroms of density maxima
        for i in tqdm(range(len(u.trajectory))):       
        # for i in tqdm(range(100)):       
            u.trajectory[i]
            radius= ' 2.5'
            atomgroup_IDS=list(u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[atom_no] +radius).indices)
            if len(atomgroup_IDS)==0:
                atomgroup_IDS=[-1]
            counting.append(atomgroup_IDS)

        ## Atom indices that appear in the atom site
        flat_list = [item for sublist in counting for item in sublist]

        atom_ID = 'a' + str(atom_no+1)
        atom_location = coords[atom_no] + g.origin
        pocket_occupation_frequency = 1 - flat_list.count(-1)/len(flat_list)   
        pocket_occupation_frequency = round(pocket_occupation_frequency,4)
        atom_information.append([atom_ID,list(atom_location),pocket_occupation_frequency])
        atom_dists.append(counting)

        ## Write data out and visualize atom sites in pdb           
        if write is True:
            data_out('atom_features/' + out_name + atom_ID + '.txt', [counting])                    
            data_out('atom_features/'+out_name+element+'AtomsSummary.txt', atom_information)                          
            write_atom_to_pdb(pdb_outname, atom_location, atom_ID, atomgroup)

            u_pdb = mda.Universe(pdb_outname)
            u_pdb.add_TopologyAttr('tempfactors')
            # Write values as beta-factors ("tempfactors") to a PDB file
            for res in range(len(atom_information)):
                atom_resid = len(u_pdb.residues) - atom_no-1 + res
                u_pdb.residues[atom_resid].atoms.tempfactors = atom_information[res][-1]
            u_pdb.atoms.write(pdb_outname)  
            
    # Add atom pocket atomIDs
    feature_names[element+'Pocket_Idx']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_Idx']= np.array(atom_dists, dtype=object)    
    
    # Add atom pocket frequencies
    feature_names[element+'Pocket_Occup']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_Occup']= np.array([atinfo[2] for atinfo in atom_information],dtype=object)

    # Add atom pocket occupancy timeseries
    feature_names[element+'Pocket_OccupDistr']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_OccupDistr']= np.array([convert_to_occ(distr, -1, water=False) for distr in atom_dists],dtype=object)
    
    # Add atom pocket locations
    feature_names[element+'Pocket_xyz']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_xyz']= np.array([atinfo[1] for atinfo in atom_information],dtype=object)
    
    # Return the dictionaries.
    return feature_names, features_data

