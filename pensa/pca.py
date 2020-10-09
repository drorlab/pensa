import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR PRINCIPAL COMPONENT ANALYSIS ---


def calculate_pca(data):
    '''Performs a PyEMMA PCA on the provided data
    
    Parameters
    ----------
    data: float array.
        Trajectory data [frames,frame_data]
    '''
    
    pca = pyemma.coordinates.pca(data.T)
    
    # Plot the eigenvalues
    fig,ax = plt.subplots(1,1,figsize=[4,3],dpi=100)
    ax.plot(pca.eigenvalues[:12],'o')
    plt.show()   
    
    return pca


def pca_features(pca, features, num, threshold):
    '''Prints relevant features and plots feature correlations.
    
    Parameters
    ----------
    pca: pyemma PCA object.
        The PCA of which to plot the features.
    features: list of strings
        Features for which the PCA was performed (obtained from features object via .describe()).
    num: float.
        Number of feature correlations to plot.
    threshold: float.
        Features with a correlation above this will be printed.
    '''
    
    # Plot the highest PC correlations and print relevant features
    fig,ax = plt.subplots(num,1,figsize=[4,num*3],dpi=100,sharex=True)
    for i in range(num):
        relevant = pca.feature_PC_correlation[:,i]**2 > threshold
        print(np.array(features)[relevant])
        ax[i].plot(pca.feature_PC_correlation[:,i])
    plt.show()
    
    
def project_on_pc(data, ev_idx, pca=None):
    '''Projects a trajectory onto an eigenvector of its PCA.
    
    Parameters
    ----------
    data: float array.
        Trajectory data [frames,frame_data]
    ev_idx: int
        Index of the eigenvector to project on
    pca (opt.): pyemma PCA object.
        Pre-calculated PCA. Must be calculated for the same features (but not necessarily the same trajectory)
    '''
    
    data = data.T
    
    if pca is None:
        pca = pyemma.coordinates.pca(data) #,dim=3)

    projection = np.zeros(data.shape[0])
    for ti in range(data.shape[0]):
        projection[ti] = np.dot(data[ti],pca.eigenvectors[:,ev_idx])
        
    return projection
    

def sort_traj_along_pc(data, pca, start_frame, top, trj, out_name, num_pc=3):
    '''Sort a trajectory along given principal components.
    
    Parameters
    ----------
    data: float array.
        Trajectory data [frames,frame_data]
    pca: PCA object.
        principal compoenents information
    num_pc: int
        sort along the first num_pc principal components
    start_frame: int
        offset of the data with respect to the trajectories (defined below)
    top: string.
        reference topology for the trajectory. 
    trj: string.
        first of the trajetories from which the frames are picked. 
        Should be the same as data was from.
    out_name: string.
        core part of the name of the output files
    '''
    
    # Remember the index in the simulation (taking into account cutoff)
    oidx = np.arange(len(data.T))+start_frame

    # Define the MDAnalysis trajectories from where the frames come
    u = mda.Universe(top,trj)
    a = u.select_atoms('all')

    for evi in range(num_pc):

        # Project the combined data on the principal component
        proj = project_on_pc(data,evi,pca=pca)

        # Sort everything along the projection on th resp. PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        oidx_sort = oidx[sort_idx]

        with mda.Writer(out_name+"_pc"+str(evi)+".xtc", a.n_atoms) as W:
            for i in range(data.shape[1]):
                ts = u.trajectory[oidx_sort[i]]
                W.write(a)
                
                                   
def compare_projections(data_g, data_a, pca, num=3, saveas=None):
    '''Compare two datasets along a given principal component.
    
    Parameters
    ----------
    data_g: float array.
        Trajectory data [frames,frame_data]
    data_a: float array.
        Trajectory data [frames,frame_data]
    pca: PCA object.
        Principal components information.
    num: int.
        Number of princibal components to plot. 
    saveas: string.
        Name of the output file.
    '''
    
    fig,ax = plt.subplots(num,2,figsize=[8,3*num],dpi=100)

    for evi in range(num):
        
        proj_a = project_on_pc(data_a,evi,pca=pca)
        proj_g = project_on_pc(data_g,evi,pca=pca)

        ax[evi,0].plot(proj_a,alpha=0.5)
        ax[evi,0].plot(proj_g,alpha=0.5)

        ax[evi,1].hist(proj_a,bins=30,alpha=0.5,density=True)
        ax[evi,1].hist(proj_g,bins=30,alpha=0.5,density=True)
    
    if saveas is not None:
        fig.savefig(saveas,dpi=300)
        
        