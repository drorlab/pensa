# -*- coding: utf-8 -*-
"""
Methods to obtain a distribution for the water pockets which respresents
a combination of the water occupancy (binary variable) and the water polarisation (continuous variable).

For a water molecule to exist within a water pocket, just the oxygen must occupy the pocket. 
If there is ever an instance where two water molecules occupy the same pocket at the same time,
then the water polarisation of the molecule ID that occupies the pocket most often is used.

The methods here are based on the following paper:

    |    Neil J. Thomson, Owen N. Vickery, Callum M. Ives, Ulrich Zachariae: 
    |    Ion-water coupling controls class A GPCR signal transduction pathways. 
    |    https://doi.org/10.1101/2020.08.28.271510
    
"""

import numpy as np
from scipy import ndimage as ndi
import os
# from gridData import Grid
import MDAnalysis as mda
from MDAnalysis.analysis.density import DensityAnalysis
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align
import biotite.structure as struc
import biotite.structure.io as strucio
# from pensa import *
# from pensa.features.processing import *

# -- Processing trajectories for density analysis

def _match_sim_lengths(sim1,sim2):
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

def _copy_coords(ag):
    """
    Copy the coordinates of the frames in a universe.    

    Parameters
    ----------
    ag : Universe.atoms

    Returns
    -------
    array
        Copied atom positions.

    """
    return ag.positions.copy()

def local_maxima_3D(data, order=3):
    """
    Detects local maxima in a 3D array to obtain coordinates for density maxima. 

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
        Options are: SPC, TIP3P, TIP4P, water
    out_name : str
        Prefix for written filename. 


    """        
    if not os.path.exists('dens/'):
        os.makedirs('dens/')
    
    condition_a = mda.Universe(struc_a, xtc_a)
    condition_b = mda.Universe(struc_b, xtc_b)
   
    # # # Combine both ensembles' atoms into one universe
    Combined_conditions = mda.Merge(condition_a.atoms, condition_b.atoms)
    
    # # # Extract trajectory coordinates
    aligned_coords_a = AnalysisFromFunction(_copy_coords,
                                            condition_a.atoms).run().results
    
    aligned_coords_b = AnalysisFromFunction(_copy_coords,
                                            condition_b.atoms).run().results
    
    # # # The density needs to be formed from an even contribution of both conditions
    # # # otherwise it will be unevely biased towards one condition.
    # # # So we match the simulation lengths first
    sim1_coords, sim2_coords = _match_sim_lengths(aligned_coords_a,aligned_coords_b)

    # # # Then we merge the coordinates into one system
    merged_coords = np.hstack([sim1_coords, sim2_coords])
    # # # We load in the merged coordinated into our new universe that contains
    # # # the receptor in both conditions
    Combined_conditions.load_new(merged_coords, format=MemoryReader)    

    # # # We extract the density grid from the combined condition universe
    density_atomgroup = Combined_conditions.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run()
    D.density.convert_density(write_grid_as)
    D.density.export('dens/' + out_name + atomgroup +"_density.dx", type="double")
    
    
    
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
    if not os.path.exists('dens/'):
        os.makedirs('dens/')
    
    # # Before we extract the water densities, we need to first align the trajectories 
    # # so that we can featurize water sites in both ensembles using the same coordinates
    condition_a = mda.Universe(struc_a,xtc_a)
    condition_b = mda.Universe(struc_b,xtc_b)    
    
    align_xtc_name='dens/' + struc_a.split('/')[-1][:-4] + 'aligned.xtc'    
    
    #align condition a to condition b
    align.AlignTraj(condition_a,  # trajectory to align
                    condition_b,  # reference
                    select= 'name CA',  # selection of atoms to align
                    filename= align_xtc_name,  # file to write the trajectory to
                    match_atoms=True,  # whether to match atoms based on mass
                    ).run()    
    
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
        if not os.path.exists('dens/'):
            os.makedirs('dens/')
        D.density.convert_density(write_grid_as)
        D.density.export('dens/' + out_name + atomgroup + "_density.dx", type="double")

    return g
        
def write_atom_to_pdb(pdb_outname, atom_location, atom_ID, atomgroup):
    """
    Write a new atom to a reference structure to visualise conserved non-protein atom sites.

    Parameters
    ----------
    pdb_outname : str
        Filename of reference structure.
    atom_location : array
        (x,y,z) coordinates of the atom location with respect to the reference structure.
    atom_ID : str
        A unique ID for the atom.
    atomgroup : str
        MDAnalysis atomgroup to describe the atom.

    """
    
    ##PDB_VISUALISATION     
    ##rescursively add waters to the pdb file one by one as they are processed           
    # # Read the file into Biotite's structure object (atom array)
    atom_array = strucio.load_structure(pdb_outname)
    res_id = atom_array.res_id[-1] + 1
    # Add an HETATM
    atom = struc.Atom(
        coord = atom_location,
        chain_id = "X",
        # The residue ID is the last ID in the file +1
        res_id = res_id,
        res_name = atom_ID,
        hetero = True,
        atom_name = atomgroup,
        element = "O"
        )
    atom_array += struc.array([atom])
    # Save edited structure
    strucio.save_structure(pdb_outname, atom_array)

def data_out(filename, data):
    """
    Write out lists of data

    Parameters
    ----------
    filename : str
        Name for the written file.
    data : list of lists
        Data to be written out.

    """
    with open(filename, 'w') as output:
        for row in data:
            output.write(str(row)[1:-1] + '\n')


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

