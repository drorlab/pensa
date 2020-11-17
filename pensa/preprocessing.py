import MDAnalysis as mda
import pyemma
import numpy as np


# -- Functions to preprocess trajectories --


def range_to_string(a, b):
    """
    Provides a string with all integers in between two numbers.
    
    Args:
        a (int): first number
        b (int): last number
        
    Returns:
        string (str): string containing all int numbers from a to b.
        
    """
    r = np.arange(a, b+1)
    string = ''
    for ri in r:
        string += str(ri)
        string += ' '
    return string


def load_selection(sel_file, sel_base=''):
    """
    Loads a selection from a selection file.
    
    Args:
        sel_file (int): Name of the file with selections.
            Must contain two numbers on each line (first and last residue of this part).
        sel_base (str): the basis string for the selection. Defaults to an empty string.
    
    Returns:
        sel_string (str): A selection string that provides the residue numbers for MDAnalysis.
    
    """
    sel_string = sel_base + 'resid '
    with open(sel_file,'r') as sf:
        for line in sf.readlines():
            r = np.array(line.strip().split(' '), dtype=int)
            sel_string += range_to_string(*r)
    return sel_string


def extract_coordinates(ref, pdb, trj_list, out_name, sel_string, start_frame=0):
    """
    Extracts selected coordinates from a trajectory file.
    
    Args:
        ref (str): File name for reference topology. 
            Can read all MDAnalysis-compatible topology formats.
        pdb (str): File name for the reference PDB file.
        trj_list (list of str): File names for the input trajectory.
            Can read all MDAnalysis-compatible trajectory formats.
        out_name (str): Core of the file names for the output files.
        start_frame (int, optional): First frame to read from the trajectory.
    
    """
    # Read the reference+PDB files and extract selected parts.
    u = mda.Universe(ref,pdb)
    selection = u.select_atoms(sel_string)
    selection.write(out_name+'.pdb')
    selection.write(out_name+'.gro')
    # Read the trajectories and extract selected parts.
    with mda.Writer(out_name+'.xtc', selection.n_atoms) as W:
        for trj in trj_list:
            u = mda.Universe(ref,trj)
            selection = u.select_atoms(sel_string)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
    return


def extract_coordinates_combined(ref, trj, sel_string, out_name, start_frame=0):
    """
    Extracts selected coordinates from several trajectory files.
    
    Args:
        ref (list of str): File names for the reference topologies. 
            Can read all MDAnalysis-compatible topology formats.
        trj (list of str): File names for the input trajectories.
            Can read all MDAnalysis-compatible trajectory formats.
        out_name (str): Core of the file names for the output files.
        start_frame (int, optional): First frame to read from the trajectory.
    
    """        
    # Determine the number of atoms from the first trajectory
    u = mda.Universe(ref[0], trj[0])
    selection = u.select_atoms(sel_string[0])
    num_at = selection.n_atoms              
    # Go through trajectories and write selections
    with mda.Writer(out_name+'.xtc', num_at) as W:
        for r, t, s in zip(ref, trj, sel_string):
            print(r, t)
            print(s)
            u = mda.Universe(r, t)
            selection = u.select_atoms(s)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
    return

