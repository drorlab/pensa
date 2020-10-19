import MDAnalysis as mda
import pyemma
import numpy as np



# -- Preprocessing Trajectories --


def range_to_string(a, b):
    """Converts a tuple to a string with all integers in between"""
    
    r = np.arange(a, b+1)
    string = ''
    for ri in r:
        string += str(ri)
        string += ' '

    return string


def load_selection(sel_file, sel_base=''):
    """Load a selection from a selection string"""
    
    with open(sel_file,'r') as sf:
        for line in sf.readlines():
            r = np.array(line.strip().split(' '), dtype=int)
            sel_string += range_to_string(*r)
    
    return sel_string


def extract_coordinates(ref, pdb, trj, out_name, sel_string):
    """Extract selected coordinates from a trajectory file"""
    
    u = mda.Universe(ref,pdb)
    selection = u.select_atoms(sel_string)

    selection.write(out_name+'.pdb')
    selection.write(out_name+'.gro')

    u = mda.Universe(ref,trj)
    selection = u.select_atoms(sel_string)

    with mda.Writer(out_name+'.xtc', selection.n_atoms) as W:
        for ts in u.trajectory:
            W.write(selection)
            
    return


def extract_coordinates_combined(ref, trj, sel_string, out_name, start_frame=0):
    """Extract selected coordinates from several trajectory files"""
        
    # Determine number of atoms from first trajectory
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

