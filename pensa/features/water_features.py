# -*- coding: utf-8 -*-
"""
Methods to obtain a timeseries distribution for the water pockets which respresents
a combination of the water occupancy (binary variable) and the water polarisation (continuous variable).

Water pockets are defined as radius 3.5 Angstroms (based off of hydrogen bond lengths)
centered on the probability density maxima of waters. If there is ever an instance 
where two water molecules occupy the same pocket at the same time, then the water that 
occupies the pocket most often is used to obtain the polarisation.

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
        

def _convert_to_dipole(water_atom_positions):
    """
    Convert cartesian coordinates of water atoms into spherical coordinates of water dipole.

    Parameters
    ----------
    water_atom_positions : list
        xyz coordinates for water oxygen atom.

    Returns
    -------
    psi : float
        Psi value (spherical coordinates) for water orientation.
    phi : float
        Phi value (spherical coordinates) for water orientation.

    """

    ## Coordinates for water atoms
    Ot0 = np.array(water_atom_positions[0])
    H1t0 = np.array(water_atom_positions[1])
    H2t0 = np.array(water_atom_positions[2])
    
    ## Dipole vector
    dipVector0 = (H1t0 + H2t0) * 0.5 - Ot0
    x_axis=dipVector0[0]
    y_axis=dipVector0[1]
    z_axis=dipVector0[2]
    
    ## Convert to spherical coordinates
    ## radians
    psi=np.arctan2(y_axis,x_axis)
    phi=np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2)))    
    ## degrees
    # psi=math.degrees(np.arctan2(y_axis,x_axis))
    # phi=math.degrees(np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2))))   

    return psi, phi


def get_water_features(structure_input, xtc_input, atomgroup, top_waters=10, 
                       grid_input=None, write=None, write_grid_as=None, out_name=None):
    """
    Featurize water pockets for the top X most probable waters (top_waters).

    Parameters
    ----------
    structure_input : str
        File name for the reference file (PDB or GRO format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    top_waters : int, optional
        Number of waters to featurize. The default is 10.
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
        if not os.path.exists('water_features/'):
            os.makedirs('water_features/')
        protein = u.select_atoms("protein")
        pdb_outname = 'water_features/' + out_name + "_WaterSites.pdb"
        u.trajectory[0]
        protein.write(pdb_outname)
        if grid_input is None:
            g = get_grid(u, atomgroup, write_grid_as,  out_name)           
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
    water_information=[]
    water_dists=[]

    if top_waters > len(coords):
        top_waters = len(coords)  


    print('\n')
    print('Featurizing ',top_waters,' Waters')
    for wat_no in range(top_waters):
        print('\n')
        print('Water no: ',wat_no+1)
        print('\n')
        philist=[]
        psilist=[]

        ## Find all water atoms within 3.5 Angstroms of density maxima
        counting=[]
        for frame_no in tqdm(range(len(u.trajectory))):       
        # for frame_no in tqdm(range(100)):       
            u.trajectory[frame_no]
            radius = ' 3.5'
            atomgroup_IDS = u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[wat_no] + radius).indices
            counting.append(atomgroup_IDS)
            
        ## Water atom indices that appear in the water site
        flat_list = [item for sublist in counting for item in sublist]
        
        ## Extract water orientation timeseries
        for frame_no in tqdm(range(len(u.trajectory))):   
        # for frame_no in tqdm(range(100)):       
            u.trajectory[frame_no]
            waters_resid=counting[frame_no]
            if len(waters_resid)==1:        
                ## (x,y,z) positions for the water oxygen at trajectory frame_no
                water_atom_positions = [list(pos) for pos in u.select_atoms('byres index ' + str(waters_resid[0])).positions]
                psi, phi = _convert_to_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ## Featurize water with highest pocket occupation (if multiple waters in pocket)
            elif len(waters_resid)>1:
                freq_count=[]
                for ID in waters_resid:
                    freq_count.append([flat_list.count(ID),ID])
                freq_count.sort(key = lambda x: x[0])
                water_atom_positions = [list(pos) for pos in u.select_atoms('byres index ' + str(freq_count[-1][1])).positions]
                psi, phi = _convert_to_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ## 10000.0 = no waters bound
            elif len(waters_resid)<1:
                psilist.append(10000.0)
                philist.append(10000.0)

        water_out = [psilist, philist]
        water_dists.append(water_out)        
        water_ID = "O" + str(wat_no+1)
        water_pocket_occupation_frequency = 1 - psilist.count(10000.0)/len(psilist)    
        water_pocket_occupation_frequency = round(water_pocket_occupation_frequency,4)
        atom_location = coords[wat_no] + g.origin

        water_information.append([water_ID,list(atom_location),water_pocket_occupation_frequency])
        
        ## Write data out and visualize water sites in pdb           
        if write is True:    
            data_out('water_features/' + out_name + water_ID + '.txt', water_out)                    
            data_out('water_features/' + out_name + 'WatersSummary.txt', water_information)                          
            write_atom_to_pdb(pdb_outname, atom_location, water_ID, atomgroup)
            u_pdb = mda.Universe(pdb_outname)
            u_pdb.add_TopologyAttr('tempfactors')
            # Write values as beta-factors ("tempfactors") to a PDB file
            for res in range(len(water_information)):
                #scale the water resid by the starting resid
                water_resid = len(u_pdb.residues) - wat_no-1 + res
                u_pdb.residues[water_resid].atoms.tempfactors = water_information[res][-1]
            u_pdb.atoms.write(pdb_outname)
        
    # Add water pocket orientations
    feature_names['WaterPocket_Distr']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_Distr']= np.array(water_dists, dtype=object)
    
    occup=[watinf[2] for watinf in water_information]
    # Add water pocket occupancies
    feature_names['WaterPocket_Occup']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_Occup']= np.array(occup, dtype=object)
    
    occup_distr=[[convert_to_occ(distr[0], 10000.0)] for distr in water_dists]
    # Add water pocket occupancy timeseries
    feature_names['WaterPocket_OccupDistr']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_OccupDistr']= np.array(occup_distr, dtype=object)
    
    loc=[watinf[1] for watinf in water_information]
    # Add water pocket locations
    feature_names['WaterPocket_xyz']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_xyz']= np.array(loc, dtype=object)
    
    # Return the dictionaries.
    return feature_names, features_data
            
