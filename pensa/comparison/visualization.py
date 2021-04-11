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
        

def distances_visualization(dist_names, dist_diff, plot_filename, 
                            vmin=None, vmax=None, verbose=True, cbar_label=None):
    """
    Visualizes features per residue as plot and in PDB files, assuming values from 0 to 1. 
    
    Parameters
    ----------
        dist_names : str array
            Names of the features in PyEMMA nomenclaturre (contain residue IDs).
        dist_diff : float array
            Data for each distance feature.
        plot_filename : str
            Name of the file for the plot.
        vmin : float, optional
            Minimum value for the heat map.
        vmax : float, optional
            Maximum value for the heat map.
        verbose : bool, optional
            Print numbers of first and last residue. Defaults to True.
        cbar_label : str, optional
            Label for the color bar.
        
    Returns
    -------
        diff : float array
            Distance matrix.
         
    """
    # Calculate the distance Matrix
    firstres = int(dist_names[0].split(' ')[-1])
    lastres  = int(dist_names[-1].split(' ')[-1])
    if verbose:
        print('Plotting distance matrix')
        print('first res:', firstres, ', last res:', lastres)
    size = lastres-firstres+2
    diff = np.zeros([size,size])
    for n,name in enumerate(dist_names):
        splitname = name.split(' ')
        resi,resj = int(splitname[2]),int(splitname[6])
        i = resi - firstres
        j = resj - firstres
        diff[i,j] = dist_diff[n]
        diff[j,i] = dist_diff[n]  
    # Plot it as a heat map
    fig,ax = plt.subplots(1,1,figsize=[6,4],dpi=300)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(50-firstres,lastres-firstres+1,50))
    ax.set_yticks(np.arange(50-firstres,lastres-firstres+1,50))
    ax.set_xticklabels(np.arange(50,lastres+1,50))
    ax.set_yticklabels(np.arange(50,lastres+1,50))
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('residue number')
    ax.set_ylabel('residue number')
    fig.colorbar(img, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(plot_filename,dpi=300)  
    return diff


