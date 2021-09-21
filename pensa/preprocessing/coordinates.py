import MDAnalysis as mda
import pyemma
import numpy as np
import os
import requests
from MDAnalysis.analysis import align


# -- Functions to preprocess trajectories --


def extract_coordinates(top, pdb, trj_list, out_name, sel_string, start_frame=0,
                        rename_segments=None, residues_offset=0 ):
    """
    Extracts selected coordinates from a trajectory file.
    
    Parameters
    ----------
        top : str
            File name for topology. 
            Can read all MDAnalysis-compatible topology formats.
        pdb : str
            File name for the reference PDB file.
        trj_list : list of str
            File names for the input trajectory.
            Can read all MDAnalysis-compatible trajectory formats.
        out_name : str
            Core of the file names for the output files.
        start_frame : int, optional
            First frame to read from the trajectory.
    
    """
    # Read the topology+PDB files and extract selected parts.
    u = mda.Universe(top,pdb)
    u.residues.resids -= residues_offset
    selection = u.select_atoms(sel_string)
    num_at = selection.n_atoms
    if rename_segments is not None:
        for s in selection.segments: 
            s.segid = rename_segments
    selection.write(out_name+'.pdb')
    selection.write(out_name+'.gro')
    # Read the trajectories and extract selected parts.
    with mda.Writer(out_name+'.xtc', selection.n_atoms) as W:
        for trj in trj_list:
            u = mda.Universe(top,trj)
            u.residues.resids -= residues_offset
            selection = u.select_atoms(sel_string)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
    return num_at


def extract_coordinates_combined(top, trj, sel_string, out_name, start_frame=0, verbose=False):
    """
    Extracts selected coordinates from several trajectory files.
    
    Parameters
    ----------
        top : list of str 
            File names for the topologies. 
            Can read all MDAnalysis-compatible topology formats.
        trj : list of str
            File names for the input trajectories.
            Can read all MDAnalysis-compatible trajectory formats.
        out_name : str
            Core of the file names for the output files.
        start_frame : int, optional
            First frame to read from the trajectory.
    
    """        
    # Determine the number of atoms from the first trajectory
    u = mda.Universe(top[0], trj[0])
    selection = u.select_atoms(sel_string[0])
    num_at = selection.n_atoms              
    # Go through trajectories and write selections
    with mda.Writer(out_name+'.xtc', num_at) as W:
        for r, t, s in zip(top, trj, sel_string):
            print(r, t)
            if verbose: print(s)
            u = mda.Universe(r, t)
            selection = u.select_atoms(s)
            for ts in u.trajectory[start_frame:]:
                W.write(selection)
    return num_at


def merge_coordinates(top_files, trj_files, out_name, segid=None):
    """
    Merge several trajectories of the same system or system part.
    All trajectories must be (at least) as long as the first one.
    
    Parameters
    ----------
        top_files : str[]
            List of input topology files.
        trj_files : str[]: 
            List of input trajectory files.
        out_name : str
            Name of the output files (without ending).
        segid : str, optional
            Value to overwrite the segment ID. Defaults to None.
    
    Returns
    -------
        univ : obj
            MDAnalysis universe of the merged system.

    """
    num_parts = len(top_files)
    assert num_parts == len(trj_files)
    # Create an array of universes
    u = [ mda.Universe(top_files[i],trj_files[i]) for i in range(num_parts) ]
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


def align_coordinates(top, pdb, trj_list, out_name, sel_string='all', start_frame=0):
    """
    Aligns selected coordinates from a trajectory file.

    Parameters
    ----------
        top : str
            File name for the topology.
            Can read all MDAnalysis-compatible topology formats.
        pdb : str
            File name for the reference PDB file.
        trj_list : list of str
            File names for the input trajectory.
            Can read all MDAnalysis-compatible trajectory formats.
        out_name : str
            Core of the file names for the output files
        start_frame : int, optional
            First frame to read from the trajectory. 
    """
    # Read the topology+PDB files and align selected parts.
    u = mda.Universe(top, pdb)
    for trj in trj_list:
        mobile = mda.Universe(top, trj)
        #mobile.trajectory = mobile.trajectory[start_frame:]
        alignment = align.AlignTraj(mobile, u, select=sel_string, filename=f'{out_name}.xtc')
        alignment.run()


def sort_coordinates(values, top_name, trj_name, out_name, start_frame=0, verbose=False):
    """
    Sort coordinate frames along corresponding values.
    
    Parameters
    ----------
    values: float array
        Values along which to sort the trajectory.
    top_name: str
        Topology for the trajectory. 
    trj_name: str
        Trajetory from which the frames are picked. 
        Usually the same as the values are from.
    out_name: str
        Name of the output trajectory (usual format is .xtc).
    start_frame: int
        Offset of the data with respect to the trajectories.
        
    Returns
    -------
        data_sort: float array
            Sorted values of the input data.
        sort_idx: float array
            Sorted indices of the values.
        oidx_sort: float array
            Sorted indices of the trajectory.
            
    """
    # Remember the index in the simulation (taking into account offset)
    oidx = np.arange(len(values)) + start_frame
    # Define the MDAnalysis trajectory from where the frames come
    if verbose: print('Loading:', top_name, trj_name)
    u = mda.Universe(top_name, trj_name)
    if verbose: 
        print('Trajectory length:', len(u.trajectory))
        print('Number of values: ', len(values))
        print('Trajectory offset:', start_frame)
    a = u.select_atoms('all')
    # Sort everything along the projection on the values
    sort_idx  = np.argsort(values)
    data_sort = values[sort_idx]
    oidx_sort = oidx[sort_idx]
    # Write out sorted trajectory
    with mda.Writer(out_name, a.n_atoms) as W:
        for i in range(len(values)):
            ts = u.trajectory[oidx_sort[i]]
            W.write(a)
    return data_sort, sort_idx, oidx_sort


def merge_and_sort_coordinates(values, top_names, trj_names, out_name, start_frame=0, verbose=False):
    """
    Write multiple trajectories of coordinate frames into one trajectory, sorted along corresponding values.
    
    Parameters
    ----------
    values: list of float arrays
        Values along which to sort the trajectory.
    top_names: list of str
        topology for the trajectory. 
    trj_names: list of str
        Trajetory from which the frames are picked. 
        Usually the same as the values are from.
    out_name: str
        Name of the output trajectory (usual format is .xtc).
    start_frame: int or list of int
        Offsets of the data with respect to the trajectories.
        Defaults to zero.
        
    Returns
    -------
        data_sort: float array
            Sorted values of the input data.
        sort_idx: float array
            Sorted indices of the values.
        oidx_sort: float array
            Sorted indices of the trajectory.
            
    """
    assert type(values) == list and type(top_names) == list and type(trj_names) == list
    # Get some stats
    num_traj = len(values)
    num_frames = [len(val) for val in values]
    # Number of values must be consistent with topologies and trajectories
    assert num_traj == len(top_names) 
    assert num_traj == len(trj_names)
    # Set offset if not provided
    assert type(start_frame) == int or len(start_frame) == num_traj
    if type(start_frame) == int:
       start_frame *= np.ones(num_traj)
       start_frame = start_frame.tolist()
    # Make sure the start indices are integers (MDA does not accept floats for indexing a trajectory) 
    start_frame = [int(sf) for sf in start_frame]
    
    # Combine the input data
    data = np.concatenate(values)
    # Remember which simulation the data came frome
    cond = np.concatenate([i*np.ones(num_frames[i], dtype=int) for i in range(num_traj)])
    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(num_frames[i], dtype=int) + start_frame[i] for i in range(num_traj)])
    assert type(oidx[0]==int)
    
    # Define the MDAnalysis trajectories from where the frames come
    univs = []
    atoms = []
    for j in range(num_traj):
        u = mda.Universe(top_names[j],trj_names[j])
        if verbose: print('Length of trajectory',len(u.trajectory))
        univs.append(u)
        atoms.append(u.select_atoms('all'))
        # Make sure the numbers of atoms are the same in each trajectory
        assert atoms[j].n_atoms == atoms[0].n_atoms

    # Sort everything along the data
    sort_idx  = np.argsort(data)
    data_sort = data[sort_idx]
    cond_sort = cond[sort_idx]
    oidx_sort = oidx[sort_idx]
    
    # Write the trajectory, ordered along the PC
    with mda.Writer(out_name, atoms[0].n_atoms) as W:
        for i in range(len(data)):
            j = cond_sort[i]
            o = oidx_sort[i]
            uj = univs[j] 
            ts = uj.trajectory[o]
            W.write(atoms[j])
          
    return data_sort, sort_idx, oidx_sort  
    
