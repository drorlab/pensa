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
import matplotlib.pyplot as plt
import math
import re
from tqdm import tqdm
import os
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb
from tempfile import gettempdir, NamedTemporaryFile
       
"""    
FUNCTIONS
"""

## convert the cosine of the dipole moment into spherical coordinates 
def get_dipole(water_atom_positions):
    """
    

    Parameters
    ----------
    water_atom_positions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ##obtaining the coordinates for each of the individual atoms
    Ot0 = water_atom_positions[::3]
    H1t0 = water_atom_positions[1::3]
    H2t0 = water_atom_positions[2::3]
#    print(Ot0,H1t0,H2t0)
    ##finding the dipole vector
    dipVector0 = (H1t0 + H2t0) * 0.5 - Ot0
    
    ##getting the dot product of the dipole vector about each axis
    unitdipVector0 = dipVector0 / \
        np.linalg.norm(dipVector0, axis=1)[:, None]
#    print(unitdipVector0)
    x_axis=unitdipVector0[0][0]
    y_axis=unitdipVector0[0][1]
    z_axis=unitdipVector0[0][2]
    
    ##converting the cosine of the dipole about each axis into phi and psi
    # psi=math.degrees(np.arctan2(y_axis,x_axis))
    # phi=math.degrees(np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2))))   

    # if psi < 0:
    #     psi+=360
    # if phi < 0:
    #     phi+=360
        
        
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
    

def get_water_features(structure_input, xtc_input, atomgroup=None,
                       grid_input=None, top_waters=None, write=None, pdb_vis=True):
    """
    

    Parameters
    ----------
    structure_input : str
        .GRO file. For visualisation of the waters,
        the structure must be the reference file that the trajectory was fit to.
    xtc_input : TYPE
        Can be .trr pf .xtc.
    atomgroup : TYPE, optional
        The atomgroup name in the .GRO file. The default is "OW".
    water_model : TYPE, optional
        The type of water model used in the simulation. The default is "TIP3P".
    grid_input : TYPE, optional
        A .dx grid file to be input for water featurisation. 
        The default is None and a grid file is generated and written for viewing.
    top_waters: TYPE, optional
        Extract X waters with the highest densities.
        The default is 10
    write : TYPE, optional
        Write out the water features into sub-directory. The default is None.
    pdb_vis : TYPE, optional
        Visualise the waters in a pdb file with b-factors 
        corresponding to the probability of finding water there. The default is True.

    Returns
    -------
    water_frequencies : TYPE
        DESCRIPTION.

    """

    
    if pdb_vis is True:
        u_pdb = mda.Universe(structure_input)
        protein = u_pdb.select_atoms("protein")
        pdb_outname = structure_input[0:-4]+"_WaterSites.pdb"
        protein.write(pdb_outname)
    


        
    ## by default make this average_probability_density
    
    u = mda.Universe(structure_input, xtc_input)
    # ## The density will be obtained from the universe which depends on the .xtc and .gro
    if atomgroup is None:
        atomgroup = "OW"

    if grid_input is None:
        density_atomgroup = u.select_atoms("name " + atomgroup)
        D = DensityAnalysis(density_atomgroup, delta=1.0)
        D.run()
        D.density.convert_density("TIP3P")
        D.density.export(atomgroup + "_density.dx", type="double")
        grid_input = atomgroup + "_density.dx"

    g = Grid(grid_input)
    # ##converting the density to a probability
    sol_number = len(u.select_atoms('name ' + atomgroup))
    grid_data = np.array(g.grid)*sol_number/np.sum(np.array(g.grid))
    
    ##mask all probabilities below the average prob in the whole box
    # average_probability_density_box = sol_number/np.product(grid_data.shape)
    ##mask all probabilities below the average water probability
    average_probability_density = sol_number/sum(1 for i in grid_data.flat if i)
    print(average_probability_density)

    
    # print(average_probability_density)
    ##mask all grid centers with density less than threshold density
    grid_data[grid_data <= average_probability_density] = 0.0
    
    
    xyz, val = local_maxima_3D(grid_data)
    ##negate the array to get descending order from most prob to least prob
    val_sort = np.argsort(-1*val.copy())
    values = [val[i] for i in val_sort]    
    coords = [xyz[i] for i in val_sort]    
    print(values[0:100])    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    
    water_frequencies=[]
        
    if top_waters is None:
        top_waters = len(coords)  
    elif top_waters > len(coords):
        top_waters = len(coords)  

        
    print('Featurizing ',top_waters,' Waters')
    for wat_no in range(top_waters):
        print('\n')
        print('Water no: ',wat_no)
        print('\n')
        philist=[]
        psilist=[]

        ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
        #print('extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound')
        counting=[]
        for i in tqdm(range(len(u.trajectory))):       
        # for i in tqdm(range(100)):       
            u.trajectory[i]
            ##list all water resids within sphere of radius 2 centered on water prob density maxima
            atomgroup_IDS=list(u.select_atoms('name ' + atomgroup + ' and point ' + maxdens_coord_str[wat_no] +' 3.5').residues.resids)
            ##select only those resids that have all three atoms within the water pocket
            multi_waters_id=[]            
            for i in atomgroup_IDS:
                if len(u.select_atoms('resid ' + str(i) + ' and point '+ maxdens_coord_str[wat_no]+' 3.5'))==3:
                    multi_waters_id.append(i)
            counting.append(multi_waters_id)
        
        # ##making a list of the water IDs that appear in the simulation in that pocket (no dups)
        flat_list = [item for sublist in counting for item in sublist]
        
        ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
        # for i in tqdm(range(100)):       
        for i in tqdm(range(len(u.trajectory))):       
            u.trajectory[i]
            waters_resid=counting[i]
            ##extracting the water coordinates for inside the pocket
            if len(waters_resid)==1:        
                ##(x,y,z) positions for the water atom (residue) at frame i
                water_indices=u.select_atoms('resid ' + str(waters_resid[0])).indices
                water_atom_positions=u.trajectory[i].positions[water_indices]
                #print(water_atom_positions)
                psi, phi = get_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ##if multiple waters in pocket then use water with largest frequency of pocket occupation
            elif len(waters_resid)>1:
                freq_count=[]
                for ID in waters_resid:
                    freq_count.append([flat_list.count(ID),ID])
                freq_count.sort(key = lambda x: x[0])
                ##(x,y,z) positions for the water atom (residue) at frame i
                water_indices=u.select_atoms('resid ' + str(freq_count[-1][1])).indices
                water_atom_positions=u.trajectory[i].positions[water_indices]
                psi, phi = get_dipole(water_atom_positions)
                psilist.append(psi)
                philist.append(phi)
            ##if there are no waters bound then append these coordinates to identify 
            ##a separate state
            elif len(waters_resid)<1:
                psilist.append(10000.0)
                philist.append(10000.0)

        water_out = [psilist, philist]        
        water_ID = atomgroup + chr(ord('`')+wat_no+1)
        water_pocket_occupation_frequency = 1 - psilist.count(10000.0)/len(psilist)    
        atom_location = coords[wat_no] + g.origin
        # print(water_ID, water_pocket_occupation_frequency)

        water_frequencies.append([water_ID,atom_location,water_pocket_occupation_frequency])

        ##WRITE OUT WATER FEATURES INTO SUBDIRECTORY
        if write is True:
            if not os.path.exists('water_features/'):
                os.makedirs('water_features/')
            filename= 'water_features/' + water_ID + '.txt'
            with open(filename, 'w') as output:
                for row in water_out:
                    output.write(str(row) + '\n')

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
            u_pdb.residues[-1*res].atoms.tempfactors = water_frequencies[-1*res][-1]
        u_pdb.atoms.write(pdb_outname)

    if write is True:
        filename= 'water_features/WaterPocketFrequencies.txt'
        with open(filename, 'w') as output:
            for row in water_frequencies:
                output.write(str(row) + '\n')
            
    return water_frequencies


   

# water_frequencies=get_water_features(structure_input = "na4dkldens.gro", 
#                                     xtc_input = "trajforh2ona4dkl.xtc",
#                                     grid_input = "OW_density.dx",
#                                     # top_waters = 10,
#                                     write=True)

