import numpy as np
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt
from pensa.preprocessing import sort_coordinates, merge_and_sort_coordinates


def project_on_eigenvector(data, ev_idx, ana):
    """
    Projects a trajectory onto an eigenvector of its PCA/tICA.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        ev_idx : int
            Index of the eigenvector to project on (starts with zero). 
        ana : PCA or tICA obj
            Information of pre-calculated PCA or tICA.
            Must be calculated for the same features (but not necessarily the same trajectory).
    
    Returns
    -------
        projection : float array
            Value along the PC for each frame.
        
    """
    # Project the features onto the components
    projection = np.zeros(data.shape[0])
    for ti in range(data.shape[0]):
        projection[ti] = np.dot(data[ti], ana.eigenvectors[:,ev_idx])
    # Return the value along the PC for each frame  
    return projection
    
    
def compare_projections(data_a, data_b, ana, num=3, saveas=None, label_a=None, label_b=None):
    """
    Compare two datasets along the components of a PCA or tICA.
    
    Parameters
    ----------
        data_a : float array
            Trajectory data [frames,frame_data].
        data_b : float array
            Trajectory data [frames,frame_data].
        ana : PCA or tICA object
            Components analysis information.
        num : int
            Number of components to plot. 
        saveas : str, optional
            Name of the output file.
        label_a : str, optional
            Label for the first dataset.
        label_b : str, optional
            Label for the second dataset.

    Returns
    -------
        projections : list of float arrays
            Projections of the trajectory on each component.
                    
    """
    if label_a is not None and label_b is not None:
        labels = [label_a, label_b]
    else:
        labels = None
    projections = compare_mult_projections([data_a, data_b], ana, num=num, saveas=saveas, labels=labels, colors=None)
    return projections
    
    
def compare_mult_projections(data, ana, num=3, saveas=None, labels=None, colors=None):
    """
    Compare multiple datasets along the components of a PCA or tICA.
    
    Parameters
    ----------
        data : list of float arrays
            Data from multiple trajectories [frames,frame_data].
        ana : PCA or tICA object
            Components analysis information.
        num : int
            Number of principal components to plot. 
        saveas : str, optional
            Name of the output file.
        labels : list of str, optional
            Labels for the datasets. If provided, it must have the same length as data.
            
    Returns
    -------
        projections : list of float arrays
            Projections of the trajectory on each principal component.
        
    """
    if labels is not None:
        assert len(labels) == len(data)
    else:
        labels = [None for _ in range(len(data))]
    if colors is not None:
        assert len(colors) == len(data)
    else:
        colors = ['C%i'%num for num in range(len(data))]
    # Start the figure    
    fig,ax = plt.subplots(num, 2, figsize=[9,3*num], dpi=300)
    # Loop over components
    projections = []
    for evi in range(num):
        proj_evi = []
        for j,d in enumerate(data):
            # Calculate values along PC for each frame
            proj = project_on_eigenvector(d, evi, ana)
            # Plot the time series in the left panel
            ax[evi,0].plot(proj, alpha=0.5, 
                           label=labels[j], color=colors[j])
            # Plot the histograms in the right panel
            ax[evi,1].hist(proj, bins=30, alpha=0.5, density=True, 
                           label=labels[j], color=colors[j])
            proj_evi.append(proj)
        projections.append(proj_evi)
        # Axis labels
        ax[evi,0].set_xlabel('frame number')
        ax[evi,0].set_ylabel('PC %i'%(evi+1))            
        ax[evi,1].set_xlabel('PC %i'%(evi+1))
        ax[evi,1].set_ylabel('frequency')
        # Legend
        if labels[0] is not None:
            ax[evi,0].legend()
            ax[evi,1].legend()
    fig.tight_layout()
    # Save the figure
    if saveas is not None:
        fig.savefig(saveas, dpi=300)
    return projections
    
    
def sort_traj_along_projection(data, ana, top, trj, out_name, num_comp=3, start_frame=0):
    """
    Sort a trajectory along given principal components.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        ana : PCA or tICA obj
            Components information.
        top : str
            File name of the reference topology for the trajectory. 
        trj : str
            File name of the trajetory from which the frames are picked. 
            Should be the same as data was from.
        out_name : str
            Core part of the name of the output files
        num_comp : int, optional
            Sort along the first num_comp components.
        start_frame : int, optional
            Offset of the data with respect to the trajectories (defined below).
    
    Returns
    -------
        sorted_proj: list
            sorted projections on each component
        sorted_indices_data : list
            Sorted indices of the data array for each component
        sorted_indices_traj : list
            Sorted indices of the coordinate frames for each component

    """
    # Initialize output
    sorted_proj = []
    sorted_indices_data = []
    sorted_indices_traj = []
    # Loop through the principal components
    for evi in range(num_comp):
        # Project the combined data on the principal component
        proj = project_on_eigenvector(data, evi, ana)
        # Sort everything along the projection onto the PC
        out_xtc = out_name+"_pc"+str(evi+1)+".xtc"
        proj_sort, sort_idx, oidx_sort = sort_coordinates(proj, top, trj, out_xtc, start_frame=start_frame)
        sorted_proj.append(proj_sort)
        sorted_indices_data.append(sort_idx)
        sorted_indices_traj.append(oidx_sort)
    return sorted_proj, sorted_indices_data, sorted_indices_traj
   
