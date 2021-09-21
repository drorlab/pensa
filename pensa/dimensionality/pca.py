import numpy as np
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt
from pensa.preprocessing import sort_coordinates, merge_and_sort_coordinates
from .visualization import project_on_eigenvector, sort_traj_along_projection


# --- METHODS FOR PRINCIPAL COMPONENT ANALYSIS ---

# http://www.emma-project.org/latest/api/generated/pyemma.coordinates.pca.html


def calculate_pca(data, dim=-1):
    """
    Performs a PyEMMA PCA on the provided data.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        dim : int, optional, default = -1
            The number of dimensions (principal components) to project onto. 
            -1 means all numerically available dimensions will be used.
        
    Returns
    -------
        pca : PCA obj
            Principal components information.
        
    """
    pca = pyemma.coordinates.pca(data, dim=dim)
    return pca


def pca_eigenvalues_plot(pca, num=12, plot_file=None):
    """
    Plots the highest eigenvalues over the numberr of the principal components.
    
    Parameters
    ----------
        pca : PCA obj
            Principal components information.
        num : int, optional, default = 12
            Number of eigenvalues to plot.
        plot_file : str, optional, default = None
            Path and name of the file to save the plot.
        
    """
    # Plot eigenvalues over component numbers
    fig,ax = plt.subplots(1, 1, figsize=[4,3], dpi=300)
    componentnr = np.arange(num)+1 
    eigenvalues = pca.eigenvalues[:num]
    ax.bar(componentnr, eigenvalues)
    ax.set_xlabel('component number')
    ax.set_ylabel('eigenvalue')
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file, dpi=300)
    return componentnr, eigenvalues


def pca_features(pca, features, num, threshold, plot_file=None, add_labels=False):
    """
    Prints relevant features and plots feature correlations.
    
    Parameters
    ----------
        pca : PCA obj
            The PCA of which to plot the features.
        features : list of str
            Features for which the PCA was performed.
            (obtained from features object via .describe()).
        num : float
            Number of feature correlations to plot.
        threshold : float
            Features with a correlation above this will be printed.
        plot_file : str, optional, default = None
            Path and name of the file to save the plot.
        add_labels : bool, optional, default = False
            Add labels of the features to the x axis.
            
        
    """
    # Plot the highest PC correlations and print relevant features
    test_graph = []
    test_corr = []
    height = num*2+2 if add_labels else num*2
    fig,ax = plt.subplots(num,1,figsize=[4,height],dpi=300,sharex=True)
    for i in range(num):
        relevant = pca.feature_PC_correlation[:,i]**2 > threshold**2
        print("Features with abs. corr. above a threshold of %3.1f for PC %i:"%(threshold, i+1))
        for j, ft in enumerate(features):
            if relevant[j]:
                print(ft, "%6.3f"%(pca.feature_PC_correlation[j,i]))
                test_corr.append(pca.feature_PC_correlation[j,i])
        ax[i].bar(np.arange(len(features)), pca.feature_PC_correlation[:,i])
        ax[i].set_ylabel('corr. with PC%i'%(i+1))
        test_graph.append(pca.feature_PC_correlation[:,i])
    if add_labels:
        ax[-1].set_xticks(np.arange(len(features)))
        ax[-1].set_xticklabels(features,rotation=90)
    else:
        ax[-1].set_xlabel('feature index')
    fig.tight_layout()
    # Save the figure to a file
    if plot_file: fig.savefig(plot_file,dpi=300)
    return test_graph, test_corr
    
    
def project_on_pc(data, ev_idx, pca=None, dim=-1):
    """
    Projects a trajectory onto an eigenvector of its PCA.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        ev_idx : int
            Index of the eigenvector to project on (starts with zero). 
        pca : PCA obj, optional, default = None
            Information of pre-calculated PCA.
            Must be calculated for the same features (but not necessarily the same trajectory).
        dim : int, optional, default = -1
            The number of dimensions (principal components) to project onto.
            Only used if tica is not provided.
    Returns
    -------
        projection : float array
            Value along the PC for each frame.
        
    """
    # Perform PCA if none is provided.
    if pca is None: pca = calculate_pca(data)
    # Project the features onto the principal components.
    projection = project_on_eigenvector(data, ev_idx, pca)  
    return projection

   
def get_components_pca(data, num, pca=None, dim=-1, prefix=''):
    """
    Projects a trajectory onto the first num eigenvectors of its PCA.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        num : int
            Number of eigenvectors to project on. 
        pca : PCA obj, optional, default = None
            Information of pre-calculated PCA.
            Must be calculated for the same features (but not necessarily the same trajectory).
        dim : int, optional, default = -1
            The number of dimensions (principal components) to project onto.
            Only used if tica is not provided.
        prefix : str, optional, default = ''
            First part of the component names. Second part is "PC"+<PC number>
    
    Returns
    -------
        comp_names : list
            Names/numbers of the components.
        components : float array
            Component data [frames,components]
        
    """
    # Perform PCA if none is provided
    if pca is None: calculate_pca(data) 
    # Project the features onto the principal components
    comp_names = []
    components = []
    for ev_idx in range(num):
        projection = np.zeros(data.shape[0])
        for ti in range(data.shape[0]):
            projection[ti] = np.dot(data[ti],pca.eigenvectors[:,ev_idx])
        components.append(projection)
        comp_names.append(prefix+'PC'+str(ev_idx+1))
    # Return the names and data  
    return comp_names, np.array(components).T
     

def sort_traj_along_pc(data, top, trj, out_name, pca=None, num_pc=3, start_frame=0):
    """
    Sort a trajectory along principal components.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data].
        top : str
            File name of the reference topology for the trajectory. 
        trj : str
            File name of the trajetory from which the frames are picked. 
            Should be the same as data was from.
        out_name : str
            Core part of the name of the output files
        pca : PCA obj, optional, default = None
            Principal components information.
            If none is provided, it will be calculated.
            Defaults to None.
        num_pc : int, optional, default = 3
            Sort along the first num_pc principal components.
        start_frame : int, optional, default = 0
            Offset of the data with respect to the trajectories (defined below).
    
    Returns
    -------
        sorted_proj: list
            sorted projections on each principal component
        sorted_indices_data : list
            Sorted indices of the data array for each principal component
        sorted_indices_traj : list
            Sorted indices of the coordinate frames for each principal component

    """
    # Calculate the principal components if they are not given.
    if pca is None: calculate_pca(data, dim=num_pc)
    # Sort the trajectory along them.
    sorted_proj, sorted_indices_data, sorted_indices_traj = sort_traj_along_projection(
        data, pca, top, trj, out_name, num_comp=num_pc, start_frame = start_frame
        )
    return sorted_proj, sorted_indices_data, sorted_indices_traj


def sort_trajs_along_common_pc(data_a, data_b, top_a, top_b, trj_a, trj_b, out_name, num_pc=3, start_frame=0):
    """
    Sort two trajectories along their most important common principal components.
    
    Parameters
    ----------
        data_a : float array
            Trajectory data [frames,frame_data].
        data_b : float array
            Trajectory data [frames,frame_data].
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
        num_pc : int, optional, default = 3
            Sort along the first num_pc principal components.
        start_frame : int or list of int, default = 0
            Offset of the data with respect to the trajectories.

    Returns
    -------
        sorted_proj: list
            sorted projections on each principal component
        sorted_indices_data : list
            Sorted indices of the data array for each principal component
        sorted_indices_traj : list
            Sorted indices of the coordinate frames for each principal component
                
    """
    sorted_proj, sorted_indices_data, sorted_indices_traj = sort_mult_trajs_along_common_pc(
        [data_a, data_b], [top_a, top_b], [trj_a, trj_b], out_name, num_pc=num_pc, start_frame = start_frame
        )
    return sorted_proj, sorted_indices_data, sorted_indices_traj


def sort_mult_trajs_along_common_pc(data, top, trj, out_name, num_pc=3, start_frame=0):
    """
    Sort multiple trajectories along their most important common principal components.

    Parameters
    ----------
        data : list of float arrays
            List of trajectory data arrays, each [frames,frame_data].
        top : list of str
            Reference topology files.
        trj : list of str
            Trajetories from which the frames are picked.
            trj[i] should be the same as data[i] was from.
        out_name : str
            Core part of the name of the output files.
        num_pc : int, optional, default = 3
            Sort along the first num_pc principal components.
        start_frame : int or list of int, default = 0
            Offset of the data with respect to the trajectories.

    Returns
    -------
        sorted_proj: list
            sorted projections on each principal component
        sorted_indices_data : list
            Sorted indices of the data array for each principal component
        sorted_indices_traj : list
            Sorted indices of the coordinate frames for each principal component

    """
    num_frames = [len(d) for d in data]
    num_traj = len(data)
    if type(start_frame) == int:
        start_frame *= np.ones(num_traj)
        start_frame = start_frame.tolist()
    # Combine the input data
    all_data = np.concatenate(data,0)
    # Calculate the principal component
    pca = calculate_pca(all_data)
    # Initialize output
    sorted_proj = []
    sorted_indices_data = []
    sorted_indices_traj = []    
    # Loop over principal components.
    for evi in range(num_pc):
        # Project the combined data on the principal component
        proj = [project_on_pc(d, evi, pca=pca) for d in data]
        # Sort everything along the projection on the respective PC
        out_xtc = out_name+"_pc"+str(evi+1)+".xtc"
        proj_sort, sort_idx, oidx_sort = merge_and_sort_coordinates(
            proj, top, trj, out_xtc, start_frame=start_frame, verbose=False
            )
        sorted_proj.append(proj_sort)
        sorted_indices_data.append(sort_idx)
        sorted_indices_traj.append(oidx_sort)        
    return sorted_proj, sorted_indices_data, sorted_indices_traj
   
