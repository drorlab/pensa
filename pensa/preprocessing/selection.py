import MDAnalysis as mda
import pyemma
import numpy as np
import os
import requests



def range_to_string(a, b):
    """
    Provides a string with all integers in between two numbers.
    
    Parameters
    ----------
        a : int
            First number.
        b : int
            Last number.
        
    Returns
    -------
        string : str
            String containing all int numbers from a to b.
        
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
    
    Parameters
    ----------
        sel_file : str
            Name of the file with selections.
            Must contain two numbers on each line (first and last residue of this part).
        sel_base : str
            The basis string for the selection. Defaults to an empty string.
    
    Returns
    -------
        sel_string : str
            A selection string that provides the residue numbers for MDAnalysis.
    
    """
    sel_string = sel_base + 'resid '
    with open(sel_file,'r') as sf:
        for line in sf.readlines():
            r = np.array(line.strip().split(' '), dtype=int)
            sel_string += range_to_string(*r)
    return sel_string


