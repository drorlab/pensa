import MDAnalysis as mda
import pyemma
import numpy as np
import os
import requests


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


def extract_coordinates(ref, pdb, trj_list, out_name, sel_string, start_frame=0,
                        rename_segments=None ):
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
    if rename_segments is not None:
        for s in selection.segments: 
            s.segid = rename_segments
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


def extract_coordinates_combined(ref, trj, sel_string, out_name, start_frame=0, verbose=False):
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
            if verbose: print(s)
            u = mda.Universe(r, t)
            selection = u.select_atoms(s)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
    return


def merge_coordinates(ref_files, trj_files, out_name, segid=None):
    """
    Merges the trajectories of several different systems or system parts.
    All trajectories must be (at least) as long as the first one.
    
    Args:
        ref_files (str[]): list of input topology files
        trj_files (str[]): list of input trajectory files
        out_name (str): name of the output files (without ending)
        segid (str, optional): Value to overwrite the segment ID. Defaults to None.
    
    Returns:
        univ (obj): MDAnalysis universe of the merged system 
    """
    num_parts = len(ref_files)
    assert num_parts == len(trj_files)
    # Create an array of universes
    u = [ mda.Universe(ref_files[i],trj_files[i]) for i in range(num_parts) ]
    num_frames = len(u[0].trajectory)
    new_num_at = sum([len(ui.atoms) for ui in u]) 
    # Create the merged starting structure
    univ = mda.core.universe.Merge(*[ui.atoms for ui in u])
    # Give all segments the same name 
    if segid is not None: 
        univ.segments.segids = segid
    # Write the merged starting structure
    univ.atoms.write(out_name+'.gro')
    univ.atoms.write(out_name+'.pdb')
    # Merge and write trajectory frame by frame
    with mda.Writer(out_name+'.xtc', new_num_at) as W:
        for f in range(num_frames):
            # Set all universes to the current timesteps
            ts = [ ui.trajectory[f] for ui in u ]
            # Make sure the trajectories add up to the correct number of atoms
            assert sum([tsi.n_atoms for tsi in ts]) == new_num_at
            # Create a universe with coordinates from this timestep
            c = mda.core.universe.Merge(*[ui.atoms for ui in u])
            # Write this frame
            W.write(c.atoms)
    return univ


# -- Functions to download from online repositories


def download_from_gpcrmd(filename, folder):
    """
    Downloads a file from GPCRmd.
    
    Args:
        filename (str): Name of the file to download. 
            Must be a file that is in GPCRmd.
        folder (str): Target directory.
            The directory is created if it does not exist.
    """  
    print('Retrieving file',filename,'from GPCRmd.')
    url = 'https://submission.gpcrmd.org/dynadb/files/Dynamics/'
    url += filename
    req = requests.get(url, allow_redirects=True)
    out = os.path.join(folder,filename)
    os.makedirs(folder, exist_ok=True)
    open(out, 'wb').write(req.content)
    return


