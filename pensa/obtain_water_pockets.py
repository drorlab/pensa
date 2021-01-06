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

       
"""    
FUNCTIONS
"""

## convert the cosine of the dipole moment into spherical coordinates 
def get_dipole(water_atom_positions):
    
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
    psi=math.degrees(np.arctan2(y_axis,x_axis))
    phi=math.degrees(np.arccos(z_axis/(np.sqrt(x_axis**2+y_axis**2+z_axis**2))))   

    if psi < 0:
        psi+=360
    if phi < 0:
        phi+=360
    
    # radianphi=phi*math.pi/180
    # radianpsi=psi*math.pi/180
    
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


def get_water_features(grid_input, structure_input, xtc_input, atom=None, threshold_density=None):
    
    """
    Example use:
        get_water_features(grid_input = "OW_density.dx", 
                           structure_input = "protein.gro", 
                           xtc_input = "protein.xtc",
                           threshold_density = 0.1)
        
    Output:
        
        Polarisation angles are output in spherical coordinates in degrees 
        
        List (phi values), 
        List (psi values), 
        List (water pocket center coordinate and frequency of pocket occupation)  
    
    """
    
    wat_no=0
    
    
    if atom is None:
        atomgroup = "OW"
    ## by default make this average_probability_density
    
    u = mda.Universe(structure_input, xtc_input)
    # ## The density will be obtained from the universe which depends on the .xtc and .gro
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
    
    ##can be used to mask all probabilities below the average 
    average_probability_density = sol_number/np.product(grid_data.shape)
    if threshold_density is None:
        threshold_density = average_probability_density
    
    
    
    ##mask all grid centers with density less than threshold density
    grid_data[grid_data <= threshold_density] = 0.0
    
    
    coords, values = local_maxima_3D(grid_data)
    
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    
    philist=[]
    psilist=[]
    
    counting=[]
    
    ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
    #print('extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound')
    for i in tqdm(range(len(u.trajectory))):       
    
        u.trajectory[i]
        ##this is where the water pocket is defined as sphere of radius 5 
        ##centred on centre of geometry of following CA atoms
        waters_resid=u.select_atoms('resname SOL and point ' + maxdens_coord_str[wat_no]+' 5').resids
                         
        ##making a list of water residue IDs for every frame where all three atoms of the water appear in the pocket
        multi_waters_id=[]
        for elem in list(set(waters_resid)):
            if list(waters_resid).count(elem)==3:
                multi_waters_id.append(elem)
        counting.append(multi_waters_id)
    
    ##making a list of the water IDs that appear in the simulation in that pocket (no dups)
    flat_list = [item for sublist in counting for item in sublist]
    no_dups=list(set(flat_list))
    
    ###extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound
    #print('extracting (psi,phi) coordinates for each water dipole specific to the frame they are bound')
    for i in tqdm(range(len(u.trajectory))):       
        u.trajectory[i]
        waters_resid=counting[i]
        ##extracting the water coordinates based on the water that appears in that frame
        ##if there is only one water in the pocket then...
        if len(waters_resid)==1:        
            ##(x,y,z) positions for the water atom (residue) at frame i
            water_indices=u.select_atoms('resid ' + str(waters_resid[0])).indices
            water_atom_positions=u.trajectory[i].positions[water_indices]
            #print(water_atom_positions)
            psi, phi = get_dipole(water_atom_positions)
            psilist.append(psi)
            philist.append(phi)
            
        ##if there are multiple waters in the pocket then find the 
        ##water that appears in the pocket with the largest frequency and use that 
        ##ID number to get the coordinate for that frame
        elif len(waters_resid)>1:
            
            freq_count=[]
            for ID in waters_resid:
                freq_count.append([flat_list.count(ID),ID])
            freq_count.sort(key = lambda x: x[0])
            
            ##(x,y,z) positions for the water atom (residue) at frame i
            water_indices=u.select_atoms('resid ' + str(freq_count[-1][1])).indices
            water_atom_positions=u.trajectory[i].positions[water_indices]
            #print(water_atom_positions)
            psi, phi = get_dipole(water_atom_positions)
            psilist.append(psi)
            philist.append(phi)
    
    
        ##if there are no waters bound then append these coordinates to identify 
        ##a separate state
        elif len(waters_resid)<1:
            psilist.append(10000.0)
            philist.append(10000.0)
            
    # plt.figure()
    # plt.plot(np.histogram([elem for elem in philist if elem !=10000.0],bins=60)[1][0:-1],np.histogram([elem for elem in philist if elem !=10000.0],bins=60)[0],label='D2.50 + Na')
    # plt.xlabel('$\phi$')
    # plt.figure()
    # plt.plot(np.histogram([elem for elem in psilist if elem !=10000.0],bins=60)[1][0:-1],np.histogram([elem for elem in psilist if elem !=10000.0],bins=60)[0],label='D2.50 + Na')
    # plt.xlabel('$\psi$')
            
    
    # filepsi='waterdistributions/'+str(simulation_names)+str(water_names[water_pocket_number])+'_water_distspsi.txt'
    # filephi='waterdistributions/'+str(simulation_names)+str(water_names[water_pocket_number])+'_water_distsphi.txt'
    
    # np.savetxt(filepsi,np.array(psilist))
    # np.savetxt(filephi,np.array(philist))
    
    pymol_output = [maxdens_coord_str[wat_no] + str(psilist.count(10000.0)/len(psilist))]
        
    return psilist, philist, pymol_output