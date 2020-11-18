import numpy as np
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR PRINCIPAL COMPONENT ANALYSIS ---


def calculate_pca(data):
    """
    Performs a PyEMMA PCA on the provided data.
    
    Args:
        data (float array): Trajectory data [frames,frame_data].
        
    Returns:
        pca (PCA obj): Principal components information.
        
    """
    pca = pyemma.coordinates.pca(data)
    return pca


def pca_eigenvalues_plot(pca, num=12, plot_file=None):
    """
    Plots the highest eigenvalues over the numberr of the principal components.
    
    Args:
        pca (PCA obj): Principal components information.
        num (int, optional): Number of eigenvalues to plot. Defaults to 12.
        plot_file(str, optional): Path and name of the file to save the plot.
        
    """
    # Plot eigenvalues over component numbers
    fig,ax = plt.subplots(1, 1, figsize=[4,3], dpi=300)
    componentnr = np.arange(num)+1 
    eigenvalues = pca.eigenvalues[:num]
    ax.plot(componentnr, eigenvalues, 'o')
    ax.set_xlabel('component number')
    ax.set_ylabel('eigenvalue')
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file, dpi=300)
    return componentnr, eigenvalues


def pca_features(pca, features, num, threshold, plot_file=None):
    """
    Prints relevant features and plots feature correlations.
    
    Args:
        pca (PCA obj): The PCA of which to plot the features.
        features (list of str): Features for which the PCA was performed.
            (obtained from features object via .describe()).
        num (float): Number of feature correlations to plot.
        threshold (float): Features with a correlation above this will be printed.
        plot_file(str, optional): Path and name of the file to save the plot.
        
    """
    # Plot the highest PC correlations and print relevant features
    fig,ax = plt.subplots(num,1,figsize=[4,num*3],dpi=300,sharex=True)
    for i in range(num):
        relevant = pca.feature_PC_correlation[:,i]**2 > threshold**2
        print("Features with abs. corr. above a threshold of %3.1f for PC %i:"%(threshold, i+1))
        for j, ft in enumerate(features):
            if relevant[j]: print(ft, "%6.3f"%(pca.feature_PC_correlation[j,i]))
        ax[i].plot(pca.feature_PC_correlation[:,i])
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file,dpi=300)
    return 
    
    
def project_on_pc(data, ev_idx, pca=None):
    """
    Projects a trajectory onto an eigenvector of its PCA.
    
    Args:
        data (float array): Trajectory data [frames,frame_data].
        ev_idx (int): Index of the eigenvector to project on.
        pca (PCA obj, optional): Information of pre-calculated PCA. Defaults to None.
            Must be calculated for the same features (but not necessarily the same trajectory).
    
    Returns:
        projection (float array): value along the PC for each frame.
        
    """
    # Perform PCA if none is provided
    if pca is None:
        pca = pyemma.coordinates.pca(data) #,dim=3)
    # Project the features onto the principal components
    projection = np.zeros(data.shape[0])
    for ti in range(data.shape[0]):
        projection[ti] = np.dot(data[ti],pca.eigenvectors[:,ev_idx])
    # Return the value along the PC for each frame  
    return projection
    

def sort_traj_along_pc(data, pca, start_frame, top, trj, out_name, num_pc=3):
    """
    Sort a trajectory along given principal components.
    
    Args:
        data (float array): Trajectory data [frames,frame_data].
        pca (PCA obj): Principal components information.
        num_pc (int): Sort along the first num_pc principal components.
        start_frame (int): Offset of the data with respect to the trajectories (defined below).
        top (str): File name of the reference topology for the trajectory. 
        trj (str): File name of the trajetory from which the frames are picked. 
            Should be the same as data was from.
        out_name (str): core part of the name of the output files
    
    """    
    # Remember the index in the simulation (taking into account cutoff)
    oidx = np.arange(len(data))+start_frame
    # Define the MDAnalysis trajectories from where the frames come
    u = mda.Universe(top,trj)
    a = u.select_atoms('all')
    # Loop through the principal components
    for evi in range(num_pc):
        # Project the combined data on the principal component
        proj = project_on_pc(data,evi,pca=pca)
        # Sort everything along the projection onto the PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        oidx_sort = oidx[sort_idx]
        # Write the trajectory, ordered along the PC
        with mda.Writer(out_name+"_pc"+str(evi)+".xtc", a.n_atoms) as W:
            for i in range(data.shape[0]):
                ts = u.trajectory[oidx_sort[i]]
                W.write(a)
    return


def sort_trajs_along_common_pc(data_a, data_b, start_frame, top_a, top_b, trj_a, trj_b, out_name, num_pc=3):
    """
    Sort two trajectories along their most important common principal components.
    
    Args:
        data_a (float array): Trajectory data [frames,frame_data].
        data_b (float array): Trajectory data [frames,frame_data].
        start_frame (int): Offset of the data with respect to the trajectories (defined below).
        top_a (str): Reference topology for the first trajectory. 
        top_b (str): Reference topology for the second trajectory. 
        trj_a (str): First of the trajetories from which the frames are picked. 
            Should be the same as data_a was from.
        trj_b (str): Second of the trajetories from which the frames are picked. 
            Should be the same as data_b was from.
        out_name (str): Core part of the name of the output files.
    
    """
    # Combine the input data
    data = np.concatenate([data_a,data_b],0)
    # Remember which simulation the data came frome
    cond = np.concatenate([np.ones(len(data_a)), np.zeros(len(data_b))])
    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(len(data_a))+start_frame, 
                           np.arange(len(data_b))+start_frame])
    # Calculate the principal components
    pca = pyemma.coordinates.pca(data,dim=3)
    # Define the MDAnalysis trajectories from where the frames come
    ua = mda.Universe(top_a,trj_a)
    ub = mda.Universe(top_b,trj_b)
    # ... and select all atoms
    aa = ua.select_atoms('all')
    ab = ub.select_atoms('all')
    # Loop over principal components.
    for evi in range(num_pc):
        # Project the combined data on the principal component
        proj = project_on_pc(data,evi,pca=pca)
        # Sort everything along the projection on th resp. PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        cond_sort = cond[sort_idx]
        oidx_sort = oidx[sort_idx]
        # Write the trajectory, ordered along the PC
        with mda.Writer(out_name+"_pc"+str(evi)+".xtc", aa.n_atoms) as W:
            for i in range(data.shape[0]):
                if cond_sort[i] == 1: # G-protein bound
                    ts = ua.trajectory[oidx_sort[i]]
                    W.write(aa)
                elif cond_sort[i] == 0: # arrestin bound
                    ts = ub.trajectory[oidx_sort[i]]
                    W.write(ab)
    return


def compare_projections(data_a, data_b, pca, num=3, saveas=None, label_a=None, label_b=None):
    """
    Compare two datasets along a given principal component.
    
    Args:
        data_a (float array): Trajectory data [frames,frame_data]
        data_b (float array): Trajectory data [frames,frame_data]
        pca (PCA object): Principal components information.
        num (int): Number of principal components to plot. 
        saveas (str, optional): Name of the output file.
        label_a (str, optional): Label for the first dataset.
        label_b (str, optional): Label for the second dataset.
        
    """
    # Start the figure    
    fig,ax = plt.subplots(num, 2, figsize=[8,3*num], dpi=300)
    # Loop over PCs
    for evi in range(num):
        # Calculate values along PC for each frame
        proj_a = project_on_pc(data_a, evi, pca=pca)
        proj_b = project_on_pc(data_b, evi, pca=pca)
        # Plot the time series in the left panel
        ax[evi,0].plot(proj_a, alpha=0.5, label=label_a)
        ax[evi,0].plot(proj_b, alpha=0.5, label=label_b)
        # Plot the histograms in the right panel
        ax[evi,1].hist(proj_a, bins=30, alpha=0.5, density=True, label=label_a)
        ax[evi,1].hist(proj_b, bins=30, alpha=0.5, density=True, label=label_b)
        # Legend
        if label_a and label_b:
            ax[evi,0].legend()
            ax[evi,1].legend()
    # Save the figure
    if saveas is not None:
        fig.savefig(saveas, dpi=300)
    return

