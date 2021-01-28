#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:38:48 2020

@author: neil

This script is designed to obtain a distribution for the water pockets which respresents
a combination of the water occupancy (binary variable) and the water polarisation (continuous variable).
For a water molecule to exist within a water pocket, all three water atoms must occupy the pocket. 
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
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align


def copy_coords(ag):
    return ag.positions.copy()

## convert the cosine of the dipole moment into spherical coordinates 
def get_dipole(water_atom_positions):

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

    return(psi,phi)
    

# obtain  coordinates for maxima of the dens 
def local_maxima_3D(data, order=3):
    """Detects local maxima in a 3D array

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


def get_grid(structure_input, xtc_input, atomgroup, grid_wat_model, write=None):
 
    u = mda.Universe(structure_input, xtc_input)
    density_atomgroup = u.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run()
    D.density.convert_density(grid_wat_model)
    if write is not None:
        D.density.export(structure_input[:-4] + atomgroup + "_density.dx", type="double")

    g = D.density
    
    return g


##make atomgroup mandatory
def get_water_features(structure_input, xtc_input, atomgroup, grid_wat_model=None,
                       grid_input=None, top_waters=30, write=None, pdb_vis=True):
    
    u = mda.Universe(structure_input, xtc_input)

    if pdb_vis is True:
        protein = u.select_atoms("protein")
        pdb_outname = structure_input[0:-4]+"_WaterSites.pdb"
        u.trajectory[0]
        protein.write(pdb_outname)

    if grid_input is None:
        density_atomgroup = u.select_atoms("name " + atomgroup)
        # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
        D = DensityAnalysis(density_atomgroup, delta=1.0)
        D.run()
        if grid_wat_model is not None:
            D.density.convert_density(grid_wat_model)
            D.density.export(structure_input[:-4] + atomgroup + "_density.dx", type="double")
            grid_input = atomgroup + "_density.dx"
        g = D.density
    else:
        g = Grid(grid_input)  
    
    
    xyz, val = local_maxima_3D(g.grid)
    ##negate the array to get descending order from most prob to least prob
    val_sort = np.argsort(-1*val.copy())
    coords = [xyz[i] for i in val_sort]    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    water_frequencies=[]

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
        for frame_no in tqdm(range(len(u.trajectory))):       
        # for frame_no in tqdm(range(100)):       
            u.trajectory[frame_no]
            ##list all water oxygens within sphere of radius X centered on water prob density maxima
            radius = ' 3.5'
            atomgroup_IDS = u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[wat_no] + radius).indices
            counting.append(atomgroup_IDS)
            
        ##making a list of the water IDs that appear in the simulation in that pocket
        flat_list = [item for sublist in counting for item in sublist]
        
        ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
        # for frame_no in tqdm(range(100)):       
        for frame_no in tqdm(range(len(u.trajectory))):   
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
        water_ID = "O" + str(wat_no+1)
        water_pocket_occupation_frequency = 1 - psilist.count(10000.0)/len(psilist)    
        atom_location = coords[wat_no] + g.origin

        water_frequencies.append([water_ID,atom_location,water_pocket_occupation_frequency])

        ##WRITE OUT WATER FEATURES INTO SUBDIRECTORY
        if write is True:
            if not os.path.exists('water_features/'):
                os.makedirs('water_features/')
            filename= 'water_features/' + structure_input[0:-4] + water_ID + '.txt'
            with open(filename, 'w') as output:
                for row in water_out:
                    output.write(str(row)[1:-1] + '\n')

        ##PDB_VISUALISATION     
        ##rescursively add waters to the pdb file one by one as they are processed           
        if pdb_vis is True:
            # # Read the file into Biotite's structure object (atom array)
            atom_array = strucio.load_structure(pdb_outname)
            # Shifting the coordinates by the grid origin
            atom_location = coords[wat_no] + g.origin
            # Add an HETATM
            atom = struc.Atom(
                coord = atom_location,
                chain_id = "W",
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
                    
    if pdb_vis is True:
        u_pdb = mda.Universe(pdb_outname)
        u_pdb.add_TopologyAttr('tempfactors')
        # Write values as beta-factors ("tempfactors") to a PDB file
        for res in range(len(water_frequencies)):
            #scale the water resid by the starting resid
            water_resid = len(u_pdb.residues) - top_waters + res
            u_pdb.residues[water_resid].atoms.tempfactors = water_frequencies[res][2]
        u_pdb.atoms.write(pdb_outname)

    if write is True:
        filename= 'water_features/' + structure_input[0:-4] + 'WaterPocketFrequencies.txt'
        with open(filename, 'w') as output:
            for row in water_frequencies:
                output.write(str(row)[1:-1] + '\n')
            
    return water_frequencies

def get_atom_features(structure_input, xtc_input, atomgroup, element,
                     grid_input=None, top_atoms=None, write=None, pdb_vis=True,grid_write=None):

    u = mda.Universe(structure_input, xtc_input)
    
    if pdb_vis is True:
        protein = u.select_atoms("protein")
        pdb_outname = structure_input[0:-4]+"_IonSites.pdb"
        u.trajectory[0]
        protein.write(pdb_outname)
    
    ## The density will be obtained from the universe which depends on the .xtc and .gro
    if grid_input is None:
        density_atomgroup = u.select_atoms("name " + atomgroup)
        D = DensityAnalysis(density_atomgroup, delta=1.0)
        D.run()
        if grid_write is not None:
            D.density.convert_density("Angstrom^{-3}")
            D.density.export(structure_input[:-4] + atomgroup + "_density.dx", type="double")
            grid_input = atomgroup + "_density.dx"
        g = D.density
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
    coords = [xyz[i] for i in val_sort]    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    
    atom_frequencies=[]
    
    if top_atoms is None:
        top_atoms = len(coords)  
    elif top_atoms > len(coords):
        top_atoms = len(coords)  

    print('\n')
    print('Featurizing ',top_atoms,' Atoms')
    for atom_no in range(top_atoms):
        print('\n')
        print('Atom no: ',atom_no)
        print('\n')

        counting=[]
        for i in tqdm(range(len(u.trajectory))):       
        # for i in tqdm(range(100)):       
            u.trajectory[i]
            ##list all water resids within sphere of radius 2 centered on water prob density maxima
            atomgroup_IDS=list(u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[atom_no] +' 2').indices)
            ##select only those resids that have all three atoms within the water pocket
            if len(atomgroup_IDS)==0:
                atomgroup_IDS=[-1]
            counting.append(atomgroup_IDS)

        atom_ID = element + str(atom_no+1)
        pocket_occupation_frequency = 1 - counting.count(-1)/len(counting)    
        atom_location = coords[atom_no] + g.origin

        atom_frequencies.append([atom_ID,atom_location,pocket_occupation_frequency])

        ##PDB_VISUALISATION     
        ##rescursively add waters to the pdb file one by one as they are processed           
        if pdb_vis is True:
            # # Read the file into Biotite's structure object (atom array)
            atom_array = strucio.load_structure(pdb_outname)
            # Shifting the coordinates by the grid origin
            atom_location = coords[atom_no] + g.origin
            # Add an HETATM
            atom = struc.Atom(
                coord = atom_location,
                chain_id = "W",
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
            
    if pdb_vis is True:
        u_pdb = mda.Universe(pdb_outname)
        
        u_pdb.add_TopologyAttr('tempfactors')
        # Write values as beta-factors ("tempfactors") to a PDB file
        for res in range(len(atom_frequencies)):
            atom_resid = len(u_pdb.residues) - top_atoms + res
            u_pdb.residues[atom_resid].atoms.tempfactors = atom_frequencies[res][2]
        u_pdb.atoms.write(pdb_outname)

    if write is True:
        if not os.path.exists('atom_features/'):
            os.makedirs('atom_features/')
        filename= 'atom_features/PocketFrequencies.txt'
        with open(filename, 'w') as output:
            for row in atom_frequencies:
                output.write(str(row)[1:-1] + '\n')

    return atom_frequencies

