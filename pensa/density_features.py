#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:38:48 2020

@author: Neil J Thomson

This script is designed to obtain a distribution for the water pockets which respresents
a combination of the water occupancy (binary variable) and the water polarisation (continuous variable).
For a water molecule to exist within a water pocket, just the oxygen must occupy the pocket. 
If there is ever an instance where two water molecules occupy the same pocket at the same time,
then the water polarisation of the molecule ID that occupies the pocket most often is used.
"""

import MDAnalysis as mda
from MDAnalysis.analysis.density import DensityAnalysis
import numpy as np
from scipy import ndimage as ndi
from gridData import Grid
import MDAnalysis.analysis.hbonds
from tqdm import tqdm
import os
import biotite.structure as struc
import biotite.structure.io as strucio
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align
# from pensa.statesinfo import *
# this function makes sure that the two simulations are the same length
def match_sim_lengths(sim1,sim2):
    """
    Make two lists the same length by truncating the longer list to match.

    Parameters
    ----------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    Returns
    -------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    """
    if len(sim1)!=len(sim2):
        if len(sim1)>len(sim2):
            sim1=sim1[0:len(sim2)]
        if len(sim1)<len(sim2):
            sim2=sim2[0:len(sim1)]  
    return sim1, sim2
        


def get_dipole(water_atom_positions):
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

    ##obtaining the coordinates for each of the individual atoms
    Ot0 = np.array(water_atom_positions[0])
    H1t0 = np.array(water_atom_positions[1])
    H2t0 = np.array(water_atom_positions[2])
    ##finding the dipole vector
    dipVector0 = (H1t0 + H2t0) * 0.5 - Ot0
    
    x_axis=dipVector0[0]
    y_axis=dipVector0[1]
    z_axis=dipVector0[2]
    
    ##converting the cosine of the dipole about each axis into phi and psi
    # psi=math.degrees(np.arctan2(y_axis,x_axis))
    # phi=math.degrees(np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2))))   

    ## radians
    psi=np.arctan2(y_axis,x_axis)
    phi=np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2)))

    return psi, phi
    

# obtain  coordinates for maxima of the dens 
def local_maxima_3D(data, order=3):
    """
    Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values

def extract_aligned_coords(struc_a, xtc_a, struc_b, xtc_b):
    """
    Writes out combined atomgroup density for both input simulations.
    

    Parameters
    ----------
    struc_a : str
        File name for the reference file (PDB or GRO format).
    xtc_a : str
        File name for the trajectory (xtc format).
    struc_b : str
        File name for the reference file (PDB or GRO format).
    xtc_b : str
        File name for the trajectory (xtc format).

    """

    # # Before we extract the water densities, we need to first align the trajectories 
    # # so that we can featurize water sites in both ensembles using the same coordinates
    condition_a = mda.Universe(struc_a,xtc_a)
    condition_b = mda.Universe(struc_b,xtc_b)
    
    align_xtc_name= struc_a[:-4] + 'aligned.xtc'    
    
    #align condition a to condition b
    align.AlignTraj(condition_a,  # trajectory to align
                    condition_b,  # reference
                    select= 'name CA',  # selection of atoms to align
                    filename= align_xtc_name,  # file to write the trajectory to
                    match_atoms=True,  # whether to match atoms based on mass
                    ).run()
   

def copy_coords(ag):
    """
    Copy the coordinates of the frames in the aligned universe.    

    Parameters
    ----------
    ag : Universe.atoms

    Returns
    -------
    array
        Copied atom positions.

    """
    return ag.positions.copy()

    
def extract_combined_grid(struc_a, xtc_a, struc_b, xtc_b, atomgroup, write_grid_as, out_name):
    """
    Writes out combined atomgroup density for both input simulations.
    

    Parameters
    ----------
    struc_a : str
        File name for the reference file (PDB or GRO format).
    xtc_a : str
        File name for the trajectory (xtc format).
    struc_b : str
        File name for the reference file (PDB or GRO format).
    xtc_b : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    write_grid_as : str
        The water model to convert the density into. 
        Options are:
            SPC, TIP3P, TIP4P, water
    out_name : str
        Prefix for written filename. 


    """
    
   
    
    # # # re-define the condition_a universe with the aligned trajectory 
    condition_a = mda.Universe(struc_a, xtc_a)
    condition_b = mda.Universe(struc_b, xtc_b)
   
    # # # We then combine the ensembles into one universe
    Combined_conditions = mda.Merge(condition_a.atoms, condition_b.atoms)
    
    # # # Extract the coordinates from the trajectories to add to the new universe
    aligned_coords_a = AnalysisFromFunction(copy_coords,
                                            condition_a.atoms).run().results
    
    aligned_coords_b = AnalysisFromFunction(copy_coords,
                                            condition_b.atoms).run().results
    
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
    density_atomgroup = Combined_conditions.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run()
    D.density.convert_density(write_grid_as)
    D.density.export(out_name + atomgroup +"_density.dx", type="double")
    

def get_grid(u, atomgroup, write_grid_as=None, out_name=None):
    """
    Obtain the grid for atomgroup density.

    Parameters
    ----------
    u : MDAnalysis universe
        Universe to obtain density grid.
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model 
        to convert the density into. The default is None.
    out_name : str, optional
        Prefix for all written filenames. The default is None.


    Returns
    -------
    g : grid
        Density grid.

    """
 
    density_atomgroup = u.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run()
    g = D.density
    
    if write_grid_as is not None:
        D.density.convert_density(write_grid_as)
        D.density.export(out_name + atomgroup + "_density.dx", type="double")

    return g
        
def convert_to_occ(distr, unocc_no, water=True):
    """
    Convert a distribution of pocket angles and occupancies into just occupancies.

    Parameters
    ----------
    distr : list
        Distribution to convert.
    unocc_no : float
        Value that represents unoccupied in the distribution.

    Returns
    -------
    occ : list
        Distribution representing pocket occupancy.

    """
    
    occ = np.ones(len(distr))
    
    if water is True:
        for item in range(len(occ)):
            if distr[item] == unocc_no:
                occ[item] = 0
    else:
        for item in range(len(occ)):
            if distr[item][0] == unocc_no:
                occ[item] = 0
    
    return list(occ)
        
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
        if write_grid_as is None:
            print('WARNING: You cannot write grid without specifying water model.')
            print('Otions include:\nSPC\nTIP3P\nTIP4P\nwater')
    
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
        if grid_input is None and write_grid_as is not None:
            g = get_grid(u, atomgroup, write_grid_as, out_name)
    elif grid_input is None:
       g = get_grid(u, atomgroup)              
    else:
        g = Grid(grid_input)  
    
    xyz, val = local_maxima_3D(g.grid)
    ##negate the array to get descending order from most prob to least prob
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
        print('Water no: ',wat_no)
        print('\n')
        philist=[]
        psilist=[]

        ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
        counting=[]
        for frame_no in tqdm(range(100)):       
        # for frame_no in tqdm(range(len(u.trajectory))):       
            u.trajectory[frame_no]
            ##list all water oxygens within sphere of radius X centered on water prob density maxima
            ##3.5 radius based off of length of hydrogen bonds. Water can in 
            ##theory move 3.5 angstr and still maintain the same interactions
            radius = ' 3.5'
            atomgroup_IDS = u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[wat_no] + radius).indices
            counting.append(atomgroup_IDS)
            
        ##making a list of the water IDs that appear in the simulation in that pocket
        flat_list = [item for sublist in counting for item in sublist]
        
        ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
        for frame_no in tqdm(range(100)):       
        # for frame_no in tqdm(range(len(u.trajectory))):   
            u.trajectory[frame_no]
            waters_resid=counting[frame_no]
            ##extracting the water coordinates for inside the pocket
            if len(waters_resid)==1:        
                ##(x,y,z) positions for the water atom (residue) at frame i
                water_atom_positions = [list(pos) for pos in u.select_atoms('byres index ' + str(waters_resid[0])).positions]
                psi, phi = get_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ##if multiple waters in pocket then use water with largest frequency of pocket occupation
            elif len(waters_resid)>1:
                freq_count=[]
                for ID in waters_resid:
                    freq_count.append([flat_list.count(ID),ID])
                freq_count.sort(key = lambda x: x[0])
                water_atom_positions = [list(pos) for pos in u.select_atoms('byres index ' + str(freq_count[-1][1])).positions]
                psi, phi = get_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ##10000.0 = no waters bound
            elif len(waters_resid)<1:
                psilist.append(10000.0)
                philist.append(10000.0)

        water_out = [psilist, philist]
        water_dists.append(water_out)        
        water_ID = "O" + str(wat_no+1)
        water_pocket_occupation_frequency = 1 - psilist.count(10000.0)/len(psilist)    
        atom_location = list(coords[wat_no] + g.origin)

        water_information.append([water_ID,atom_location,water_pocket_occupation_frequency])

        print('Completed water no: ',wat_no)
        print(water_information[-1])
        
        ##WRITE OUT WATER FEATURES INTO SUBDIRECTORY
        if write is True:                
            filename= 'water_features/' + out_name + water_ID + '.txt'
            with open(filename, 'w') as output:
                for row in water_out:
                    output.write(str(row)[1:-1] + '\n')

            filename= 'water_features/' + out_name + 'WatersSummary.txt'
            with open(filename, 'w') as output:
                for row in water_information:
                    output.write(str(row)[1:-1] + '\n')
                    
            ##PDB_VISUALISATION     
            ##rescursively add waters to the pdb file one by one as they are processed           
            # # Read the file into Biotite's structure object (atom array)
            atom_array = strucio.load_structure(pdb_outname)
            # Shifting the coordinates by the grid origin
            atom_location = coords[wat_no] + g.origin
            # Add an HETATM
            atom = struc.Atom(
                coord = atom_location,
                chain_id = "X",
                # The residue ID is the last ID in the file +1
                res_id = atom_array.res_id[-1] + 1,
                res_name = water_ID,
                hetero = True,
                atom_name = atomgroup,
                element = "O"
                )
            atom_array += struc.array([atom])
            # Save edited structure
            strucio.save_structure(pdb_outname, atom_array)
                    
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
    features_data['WaterPocket_Distr']= np.array(water_dists)
    
    # Add water pocket occupancies
    feature_names['WaterPocket_Occup']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_Occup']= np.array([watinf[2] for watinf in water_information])
    
    # Add water pocket occupancy timeseries
    feature_names['WaterPocket_OccupDistr']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_OccupDistr']= np.array([convert_to_occ(distr[0], 10000.0) for distr in water_dists])
    
    # Add water pocket locations
    feature_names['WaterPocket_xyz']= [watinf[0] for watinf in water_information]
    features_data['WaterPocket_xyz']= np.array([watinf[1] for watinf in water_information])
    
    # Return the dictionaries.
    return feature_names, features_data
            

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
        print('Atom no: ',atom_no)
        print('\n')

        counting=[]
        # for i in tqdm(range(len(u.trajectory))):       
        for i in tqdm(range(100)):       
            u.trajectory[i]
            radius= ' 2.5'
            ##radius is based off of bond length between Na and oxygen
            ##For Ca the bond length is ~2.42 =~2.5 so the same length can be used for various ions.
            ##list all atom resids within sphere of radius 2 centered on atom prob density maxima
            atomgroup_IDS=list(u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[atom_no] +radius).indices)
            if len(atomgroup_IDS)==0:
                atomgroup_IDS=[-1]
            counting.append(atomgroup_IDS)

        atom_ID = 'a' + str(atom_no+1)

        ##making a list of the water IDs that appear in the simulation in that pocket
        flat_list = [item for sublist in counting for item in sublist]
        pocket_occupation_frequency = 1 - flat_list.count(-1)/len(flat_list)    
        atom_location = list(coords[atom_no] + g.origin)
        atom_information.append([atom_ID,atom_location,pocket_occupation_frequency])
        atom_dists.append(counting)
        
        print('Completed atom no: ',atom_no)
        print(atom_information[-1])
        ##PDB_VISUALISATION     
        ##rescursively add waters to the pdb file one by one as they are processed           
        if write is True:
            file1= 'atom_features/' + out_name + atom_ID + '.txt'
            with open(file1, 'w') as output:
                output.write(str(counting)[1:-1])
            file2= 'atom_features/'+out_name+element+'AtomsSummary.txt'
            with open(file2, 'w') as output:
                for row in atom_information:
                    output.write(str(row)[1:-1] + '\n')
                
            # # Read the file into Biotite's structure object (atom array)
            atom_array = strucio.load_structure(pdb_outname)
            # Shifting the coordinates by the grid origin
            atom_location = coords[atom_no] + g.origin
            # Add an HETATM
            atom = struc.Atom(
                coord = atom_location,
                chain_id = "X",
                # The residue ID is the last ID in the file +1
                res_id = atom_array.res_id[-1] + 1,
                res_name = atom_ID,
                hetero = True,
                atom_name = atomgroup,
                element = element
                )
            atom_array += struc.array([atom])
            # Save edited structure
            strucio.save_structure(pdb_outname, atom_array)
            u_pdb = mda.Universe(pdb_outname)
            u_pdb.add_TopologyAttr('tempfactors')
            # Write values as beta-factors ("tempfactors") to a PDB file
            for res in range(len(atom_information)):
                atom_resid = len(u_pdb.residues) - atom_no-1 + res
                u_pdb.residues[atom_resid].atoms.tempfactors = atom_information[res][-1]
            u_pdb.atoms.write(pdb_outname)
            
            
    # Add atom pocket atomIDs
    feature_names[element+'Pocket_Idx']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_Idx']= np.array(atom_dists)    
    
    # Add atom pocket frequencies
    feature_names[element+'Pocket_Occup']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_Occup']= np.array([atinfo[2] for atinfo in atom_information])

    # Add atom pocket occupancy timeseries
    feature_names[element+'Pocket_OccupDistr']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_OccupDistr']= np.array([convert_to_occ(distr, -1, water=False) for distr in atom_dists])
    
    # Add atom pocket locations
    feature_names[element+'Pocket_xyz']= [atinfo[0] for atinfo in atom_information]
    features_data[element+'Pocket_xyz']= np.array([atinfo[1] for atinfo in atom_information])
    
    # Return the dictionaries.
    return feature_names, features_data

