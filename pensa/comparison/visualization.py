import numpy as np
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
from pensa.features import *




def residue_visualization(names, data, ref_filename, pdf_filename, pdb_filename, 
                          selection='max', y_label='max. JS dist. of BB torsions', 
                          offset=0):
    """
    Visualizes features per residue as plot and in PDB files.
    Assumes values from 0 to 1. 
    
    Parameters
    ----------
        names : str array
            Names of the features in PyEMMA nomenclaturre (contain residue ID).
        data : float array
            Data to project onto the structure.
        ref_filename : str
            Name of the file for the reference structure.
        pdf_filename : str
            Name of the PDF file to save the plot.
        pdb_filename : str
            Name of the PDB file to save the structure with the values to visualize.
        selection str, default='max'
            How to select the value to visualize for each residue from all its features 
            Options: 'max', 'min'.
        y_label : str, default='max. JS dist. of BB torsions'
            Label of the y axis of the plot.
        offset : int, default=0
            Number to subtract from the residue numbers that are loaded from the reference file.
        
    Returns
    -------
        vis_resids : int array
            Residue numbers.
        vis_values : float array
            Values of the quantity to be visualized.
         
    """
    # -- INITIALIZATION --
    # Structure to use for visualization
    u = mda.Universe(ref_filename)
    u.residues.resids -= offset
    vis_resids = u.residues.resids
    # Output values
    default = 0 if selection=='max' else 1
    vis_values = default*np.ones(len(vis_resids))
    # -- VALUE ASSIGNMENT --
    for i,name in enumerate(names):
        # To each residue ...
        resid = int( name.split(' ')[-1].replace(')','') )
        index = np.where(vis_resids == resid)[0][0]
        # ... assign the difference measures of the torsion angle with the higher (or lower) value
        if selection == 'max':
            vis_values[index] = np.maximum(vis_values[index], data[i])
        elif selection == 'min':
            vis_values[index] = np.minimum(vis_values[index], data[i])
    # -- FIGURE --
    fig,ax = plt.subplots(1,1,figsize=[4,3],dpi=300)
    # Plot values against residue number
    ax.bar(vis_resids, vis_values, width=1)
    ax.set_ylim(0,1)
    # Labels
    ax.set_xlabel('residue number')
    ax.set_ylabel(y_label)
    fig.tight_layout()
    # Save the figure
    fig.savefig(pdf_filename,dpi=300)
    # -- PDB FILE --
    u.add_TopologyAttr('tempfactors')
    # Write values as beta-factors ("tempfactors") to a PDB file
    for res in range(len(vis_values)):
        u.residues[res].atoms.tempfactors = vis_values[res]
    u.atoms.write(pdb_filename)    
    return vis_resids, vis_values        
        

def pair_features_heatmap(feat_names, feat_diff, plot_filename, separator=' - ',
                          num_drop_char=0, sort_by_pos=None, numerical_sort=False,
                          vmin=None, vmax=None, symmetric=True, cbar_label=None):
    """
    Visualizes data per feature pair in a heatmap. 
    
    Parameters
    ----------
        feat_names : str array
            Names of the features in PyEMMA nomenclature (contain residue IDs).
        feat_diff : float array
            Data to be plotted for each residue-pair feature.
        plot_filename : str
            Name of the file for the plot.
        separator : str
            String that separates the two parts of the pair-type feature.
        num_drop_char : int
            Number of characters to drop at the beginning of the feature name.
            Defaults to 0.
        sort_by_pos : int 
            Position in the name of the feature part of the quantity by which it is to be sorted.
            Assumes that the name is split by ' ' (single whitespace). Counting is 0-based.
            If None, the entire name of the feature part is sorted by numpy.unique().
            Defaults to None.
        numerical_sort : bool
            If true, the position defined by 'sort_by_pos' is assumed to be an integer.
            Defaults to False.
        vmin : float, optional
            Minimum value for the heatmap.
        vmax : float, optional
            Maximum value for the heatmap.
        symmetric : bool, optional
            The matrix is symmetric and values provided only for the upper or lower triangle. 
            Defaults to True.
        cbar_label : str, optional
            Label for the color bar.
        
    Returns
    -------
        diff : float array
            Matrix with the values of the difference/divergence.
         
    """
    # Create lists of all pairs of feature parts
    part1_list = []
    part2_list = []
    for name in feat_names:
        split_name = name[num_drop_char:].split(separator)
        assert len(split_name) == 2 # TODO: add warning
        part1, part2 = split_name
        part1_list.append(part1)
        part2_list.append(part2)
    all_parts = np.unique(np.array(part1_list+part2_list))
    # Sort the list if desired
    if sort_by_pos is not None:
        if numerical_sort:
            sortpos = np.array([int(part.split(' ')[sort_by_pos]) for part in all_parts],dtype=int)
        else:
            sortpos = np.array([part.split(' ')[sort_by_pos] for part in all_parts])
        all_parts = all_parts[np.argsort(sortpos)]
    # Initialize the matrix to store the values
    size = len(all_parts)
    diff = np.zeros([size,size])
    # Write the values into the matrix
    for n,name in enumerate(feat_names):
        part1, part2 = name[num_drop_char:].split(separator)
        i = np.where(all_parts == part1)
        j = np.where(all_parts == part2)
        diff[i,j] = feat_diff[n]
        if symmetric:
            diff[j,i] = feat_diff[n]  
    # Plot it as a heat map
    fig,ax = plt.subplots(1,1,figsize=[6,4],dpi=300)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(length=0,width=0)
    ax.yaxis.set_tick_params(length=0,width=0)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(all_parts)
    ax.set_yticklabels(all_parts)
    ax.xaxis.set_label_position('top')
    fig.colorbar(img, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(plot_filename,dpi=300)  
    return diff
    
    
def resnum_heatmap(feat_names, feat_diff, plot_filename, res1_pos=2, res2_pos=6,
                   vmin=None, vmax=None, symmetric=True, verbose=False, cbar_label=None, tick_step=50):
    """
    Visualizes data per residue pair in a heatmap. 
    
    Parameters
    ----------
        feat_names : str array
            Names of the features in PyEMMA nomenclature (contain residue IDs).
        feat_diff : float array
            Data to be plotted for each residue-pair feature.
        plot_filename : str
            Name of the file for the plot.
        res1_pos : int, optional, default = 2
            Position of the 1st residue ID in the feature name when separated by ' '.
        res2_pos : int, optional, default = 6
            Position of the 2nd residue ID in the feature name when separated by ' '.
        vmin : float, optional, default = None
            Minimum value for the heatmap.
        vmax : float, optional, default = None
            Maximum value for the heatmap.
        symmetric : bool, optional, default = True
            The matrix is symmetric and values provided only for the upper or lower triangle. 
            Defaults to True.
        verbose : bool, optional, default = False
            Print numbers of first and last residue. Defaults to True.
        cbar_label : str, optional, default = None
            Label for the color bar.
        tick_step : int, optional, default = 50
            Step between two ticks on the plot axes.
        
    Returns
    -------
        diff : float array
            Matrix with the values of the difference/divergence.
         
    """
    # Find first and last residue
    rn1 = [int(fn.split(' ')[res1_pos]) for fn in feat_names]
    rn2 = [int(fn.split(' ')[res2_pos]) for fn in feat_names]
    resnums = np.concatenate([np.array(rn1,dtype=int),np.array(rn2,dtype=int)])
    first_res = resnums.min()
    last_res  = resnums.max()
    if verbose: print('first res:', first_res, ', last res:', last_res)
    # Create a 2D array with the values 
    size = last_res - first_res + 1
    diff = np.zeros([size,size])
    for n, name in enumerate(feat_names):
        splitname = name.split(' ')
        resi,resj = int(splitname[res1_pos]), int(splitname[res2_pos])
        i = resi - first_res
        j = resj - first_res
        diff[i,j] = feat_diff[n]
        if symmetric:
            diff[j,i] = feat_diff[n]  
    # Plot it as a heat map
    fig,ax = plt.subplots(1,1,figsize=[6,4],dpi=300)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position('top')
    # Find position for the first tick
    first_tick = 0
    while first_res > first_tick:
        first_tick += tick_step 
    # Ticks and labels
    ax.set_xticks(np.arange(first_tick-first_res, size, tick_step))
    ax.set_yticks(np.arange(first_tick-first_res, size, tick_step))
    ax.set_xticklabels(np.arange(first_tick, last_res+1, tick_step))
    ax.set_yticklabels(np.arange(first_tick, last_res+1, tick_step))
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('residue number')
    ax.set_ylabel('residue number')
    fig.colorbar(img, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(plot_filename,dpi=300)  
    return diff
    
    
def distances_visualization(dist_names, dist_diff, plot_filename, 
                            vmin=None, vmax=None, verbose=True, 
                            cbar_label=None, tick_step=50):
    """
    Visualizes distance features for pairs of residues in a heatmap. 
    
    Parameters
    ----------
        dist_names : str array
            Names of the distances in PyEMMA nomenclature 
            (contain residue IDs at position [2] and [6] when separated by ' ').
        dist_diff : float array
            Data for each distance feature.
        plot_filename : str
            Name of the file for the plot.
        vmin : float, optional, default = None
            Minimum value for the heatmap.
        vmax : float, optional, default = None
            Maximum value for the heatmap.
        verbose : bool, optional, default = False
            Print numbers of first and last residue. Defaults to True.
        cbar_label : str, optional, default = None
            Label for the color bar.
        tick_step : int, optional, default = 50
            Step between two ticks on the plot axes.

    Returns
    -------
        diff : float array
            Distance matrix.
         
    """
    if verbose: print('Plotting heatmap for distance features.')
    diff = resnum_heatmap(dist_names, dist_diff, plot_filename, res1_pos=2, res2_pos=6,
                          vmin=vmin, vmax=vmax, verbose=verbose, cbar_label=cbar_label,
                          tick_step=tick_step)
    return diff


