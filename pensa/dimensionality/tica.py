import numpy as np
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR TIME-LAGGED INDEPENDENT COMPONENT ANALYSIS ---
# http://emma-project.org/latest/api/generated/pyemma.coordinates.tica.html#pyemma.coordinates.tica


def calculate_tica(data):
    """
    Performs a PyEMMA TICA on the provided data.
    
    Parameters
    ----------
        data : float array
            Trajectory data. Format: [frames,frame_data].
        
    Returns
    -------
        tica : TICA obj
            Time-lagged independent component information.
        
    """
    tica = pyemma.coordinates.tica(data)
    return tica


def tica_eigenvalues_plot(tica, num=12, plot_file=None):
    """
    Plots the highest eigenvalues over the numberr of the time-lagged independent components.
    
    Parameters
    ----------
        tica : TICA obj
            Time-lagged independent components information.
        num : int, default=12
            Number of eigenvalues to plot.
        plot_file : str, optional
            Path and name of the file to save the plot.
        
    """
    # Plot eigenvalues over component numbers
    fig,ax = plt.subplots(1, 1, figsize=[4,3], dpi=300)
    componentnr = np.arange(num)+1 
    eigenvalues = tica.eigenvalues[:num]
    ax.plot(componentnr, eigenvalues, 'o')
    ax.set_xlabel('component number')
    ax.set_ylabel('eigenvalue')
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file, dpi=300)
    return componentnr, eigenvalues


def tica_features(tica, features, num, threshold, plot_file=None):
    """
    Prints relevant features and plots feature correlations.
    
    Parameters
    ----------
        tica : TICA obj
            The TICA of which to plot the features.
        features : list of str
            Features for which the TICA was performed
            (obtained from features object via .describe()).
        num : float
            Number of feature correlations to plot.
        threshold : float
            Features with a correlation above this will be printed.
        plot_file : str, optional
            Path and name of the file to save the plot.
        
    """
    # Plot the highest TIC correlations and print relevant features
    fig,ax = plt.subplots(num,1,figsize=[4,num*3],dpi=300,sharex=True)
    for i in range(num):
        relevant = tica.feature_TIC_correlation[:,i]**2 > threshold**2
        print("Features with abs. corr. above a threshold of %3.1f for TIC %i:"%(threshold, i+1))
        for j, ft in enumerate(features):
            if relevant[j]: print(ft, "%6.3f"%(tica.feature_TIC_correlation[j,i]))
        ax[i].plot(tica.feature_TIC_correlation[:,i])
        test_feature = tica.feature_TIC_correlation[:,i]
        ax[i].set_xlabel('feature index')
        ax[i].set_ylabel('correlation with TIC%i'%(i+1))
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file,dpi=300)
    return test_feature
    
    
def project_on_tic(data, ev_idx, tica=None):
    """
    Projects a trajectory onto an eigenvector of its TICA.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        ev_idx : int
            Index of the eigenvector to project on (starts with zero).
        tica : TICA obj, optional
            Information of pre-calculated TICA.
            Must be calculated for the same features (but not necessarily the same trajectory).
    
    Returns
    -------
        projection : float array
            Value along the TIC for each frame.
        
    """
    # Perform TICA if none is provided
    if tica is None:
        tica = pyemma.coordinates.tica(data) #,dim=3)
    # Project the features onto the time-lagged independent components
    projection = np.zeros(data.shape[0])
    for ti in range(data.shape[0]):
        projection[ti] = np.dot(data[ti],tica.eigenvectors[:,ev_idx])
    # Return the value along the TIC for each frame  
    return projection
    

def get_components_tica(data, num, tica=None, prefix=''):
    """
    Projects a trajectory onto the first num eigenvectors of its tICA.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        num : int
            Number of eigenvectors to project on. 
        tica : tICA obj, optional
            Information of pre-calculated tICA. Defaults to None.
            Must be calculated for the same features (but not necessarily the same trajectory).
    
    Returns
    -------
        comp_names : list
            Names/numbers of the components.
        components : float array
            Component data [frames,components]
        
    """
    # Perform tICA if none is provided
    if tica is None:
        tica = pyemma.coordinates.tica(data) 
    # Project the features onto the principal components
    comp_names = []
    components = []
    for ev_idx in range(num):
        projection = np.zeros(data.shape[0])
        for ti in range(data.shape[0]):
            projection[ti] = np.dot(data[ti],tica.eigenvectors[:,ev_idx])
        components.append(projection)
        comp_names.append(prefix+'IC'+str(ev_idx+1))
    # Return the names and data
    return comp_names, np.array(components).T
    

def sort_traj_along_tic(data, tica, start_frame, top, trj, out_name, num_tic=3):
    """
    Sort a trajectory along given time-lagged independent components.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        tica : TICA obj
            Time-lagged independent components information.
        num_tic : int
            Sort along the first num_tic time-lagged independent components.
        start_frame : int
            Offset of the data with respect to the trajectories (defined below).
        top : str
            File name of the reference topology for the trajectory. 
        trj : str
            File name of the trajetory from which the frames are picked. 
            Should be the same as data was from.
        out_name : str
            Core part of the name of the output files.
    
    """    
    # Remember the index in the simulation (taking into account cutoff)
    oidx = np.arange(len(data))+start_frame
    # Define the MDAnalysis trajectories from where the frames come
    u = mda.Universe(top,trj)
    a = u.select_atoms('all')
    # Loop through the time-lagged independent components
    for evi in range(num_tic):
        # Project the combined data on the time-lagged independent component
        proj = project_on_tic(data,evi,tica=tica)
        # Sort everything along the projection onto the TIC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        oidx_sort = oidx[sort_idx]
        # Write the trajectory, ordered along the TIC
        with mda.Writer(out_name+"_tic"+str(evi+1)+".xtc", a.n_atoms) as W:
            for i in range(data.shape[0]):
                ts = u.trajectory[oidx_sort[i]]
                W.write(a)
    return oidx_sort


def sort_trajs_along_common_tic(data_a, data_b, start_frame, top_a, top_b, trj_a, trj_b, out_name, num_tic=3):
    """
    Sort two trajectories along their most important common time-lagged independent components.
    
    Parameters
    ----------
        data_a : float array
            Trajectory data [frames,frame_data].
        data_b : float array
            Trajectory data [frames,frame_data].
        start_frame : int
            Offset of the data with respect to the trajectories (defined below).
        top_a : str
            Reference topology for the first trajectory. 
        top_b : str
            Reference topology for the second trajectory. 
        trj_a : str
            First of the trajetories from which the frames are picked. 
            Should be the same as data_a was from.
        trj_b : str
            Second of the trajetories from which the frames are picked. 
            Should be the same as data_b was from.
        out_name : str
            Core part of the name of the output files.
    
    """
    # Combine the input data
    data = np.concatenate([data_a,data_b],0)
    # Remember which simulation the data came frome
    cond = np.concatenate([np.ones(len(data_a)), np.zeros(len(data_b))])
    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(len(data_a))+start_frame, 
                           np.arange(len(data_b))+start_frame])
    # Calculate the time-lagged independent components
    tica = pyemma.coordinates.tica(data,dim=3)
    # Define the MDAnalysis trajectories from where the frames come
    ua = mda.Universe(top_a,trj_a)
    ub = mda.Universe(top_b,trj_b)
    # ... and select all atoms
    aa = ua.select_atoms('all')
    ab = ub.select_atoms('all')
    # Loop over time-lagged independent components.
    for evi in range(num_tic):
        # Project the combined data on the time-lagged independent component
        proj = project_on_tic(data,evi,tica=tica)
        # Sort everything along the projection on th resp. PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        cond_sort = cond[sort_idx]
        oidx_sort = oidx[sort_idx]
        # Write the trajectory, ordered along the PC
        with mda.Writer(out_name+"_tic"+str(evi+1)+".xtc", aa.n_atoms) as W:
            for i in range(data.shape[0]):
                if cond_sort[i] == 1: # G-protein bound
                    ts = ua.trajectory[oidx_sort[i]]
                    W.write(aa)
                elif cond_sort[i] == 0: # arrestin bound
                    ts = ub.trajectory[oidx_sort[i]]
                    W.write(ab)
    return proj, oidx_sort


def sort_mult_trajs_along_common_tic(data, start_frame, top, trj, out_name, num_tic=3):
    """
    Sort multiple trajectories along their most important common time-lagged independent components.

    Parameters
    ----------
        data : list of float arrays
            List of trajectory data arrays, each [frames,frame_data].
        start_frame : int
            Offset of the data with respect to the trajectories (defined below).
        top : list of str
            Reference topology files.
        trj : list of str
            Trajetories from which the frames are picked.
            trj[i] should be the same as data[i] was from.
        out_name : str
            Core part of the name of the output files.

    """
    num_frames = [len(d) for d in data]
    num_traj = len(data)
    # Combine the input data
    data = np.concatenate(data,0)
    # Remember which simulation the data came frome
    cond = np.concatenate([i*np.ones(num_frames[i]) for i in range(num_traj)])
    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(num_frames[i])+start_frame for i in range(num_traj)])
    # Calculate the time-lagged independent components
    tica = pyemma.coordinates.tica(data,dim=3)
    # Define the MDAnalysis trajectories from where the frames come
    univs = []
    atoms = []
    for j in range(num_traj):
        u = mda.Universe(top[j],trj[j])
        univs.append(u)
        atoms.append(u.select_atoms('all'))
    # Loop over time-lagged independent component.
    for evi in range(num_tic):
        # Project the combined data on the time-lagged independent component
        proj = project_on_tic(data,evi,tica=tica)
        # Sort everything along the projection on th resp. PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx]
        cond_sort = cond[sort_idx]
        oidx_sort = oidx[sort_idx]
        # Write the trajectory, ordered along the PC
        with mda.Writer(out_name+"_tic"+str(evi+1)+".xtc", atoms[0].n_atoms) as W:
            for i in range(data.shape[0]):
                j = cond_sort[i] 
                ts = univs[j].trajectory[oidx_sort[i]]
                W.write(atoms[j])
    return


def compare_projections_tica(data_a, data_b, tica, num=3, saveas=None, label_a=None, label_b=None):
    """
    Compare two datasets along a given time-lagged indepedent component.
    
    Parameters
    ----------
        data_a : float array
            Trajectory data [frames,frame_data]
        data_b : float array
            Trajectory data [frames,frame_data]
        tica : TICA object
            Time-lagged independent components information.
        num : int, default=3
            Number of time-lagged independent components to plot. 
        saveas : str, optional
            Name of the output file.
        label_a : str, optional
            Label for the first dataset.
        label_b : str, optional
            Label for the second dataset.
        
    """
    # Start the figure    
    fig,ax = plt.subplots(num, 2, figsize=[8,3*num], dpi=300)
    # Loop over PCs
    for evi in range(num):
        # Calculate values along TIC for each frame
        proj_a = project_on_tic(data_a, evi, tica=tica)
        proj_b = project_on_tic(data_b, evi, tica=tica)
        # Plot the time series in the left panel
        ax[evi,0].plot(proj_a, alpha=0.5, label=label_a)
        ax[evi,0].plot(proj_b, alpha=0.5, label=label_b)
        ax[evi,0].set_xlabel('frame number')
        ax[evi,0].set_ylabel('TIC %i'%(evi+1))
        # Plot the histograms in the right panel
        ax[evi,1].hist(proj_a, bins=30, alpha=0.5, density=True, label=label_a)
        ax[evi,1].hist(proj_b, bins=30, alpha=0.5, density=True, label=label_b)
        ax[evi,1].set_xlabel('TIC %i'%(evi+1))
        ax[evi,1].set_ylabel('frequency')
        # Legend
        if label_a and label_b:
            ax[evi,0].legend()
            ax[evi,1].legend()
    fig.tight_layout()
    # Save the figure
    if saveas is not None:
        fig.savefig(saveas, dpi=300)
    return
