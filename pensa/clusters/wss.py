import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt

from pensa.clusters import obtain_clusters, obtain_combined_clusters


                    

def wss_over_number_of_clusters(data, algorithm='kmeans', 
                                max_iter=100, num_repeats = 5, max_num_clusters = 12,
                                plot_file = None):
    """
    Calculates the within-sum-of-squares (WSS) for different numbers of clusters,
    averaged over several iterations.
    
    Parameters
    ----------
        data : float array
            Trajectory data [frames,frame_data]
        algorithm : string
            The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        max_iter : int, optional
            Maximum number of iterations. 
            Default: 100.
        num_repeats : int, optional
            Number of times to run the clustering for each number of clusters.
            Default: 5.
        max_num_clusters : int, optional
            Maximum number of clusters for k-means clustering. 
            Default: 12.
        plot_file : str, optional
            Name of the file to save the plot.
        
    Returns
    -------
        all_wss : float array
            WSS values for each number of clusters (starting at 2).
        std_wss : float array
            Standard deviations of the WSS.
        
    """
    
    # Initialize lists
    all_wss = []
    std_wss = []
    # Loop over the number of clusters
    for nc in range(1,max_num_clusters):
        rep_wss = []
        # Run each clustering several times.
        for repeat in range(num_repeats):
            # Get clusters and WSS for this repetition.
            cc = obtain_clusters(data, algorithm=algorithm, max_iter=max_iter, 
                                 num_clusters=nc, plot=False)
            cidx, wss, centroids = cc
            rep_wss.append(wss)
        # Calculate mean and standard deviation for this number of clusters.
        all_wss.append(np.mean(rep_wss))
        std_wss.append(np.std(rep_wss))
    
    # Plot the WSS over the number of clusters
    fig, ax = plt.subplots(1,1, figsize=[4,3], dpi=300)
    ax.errorbar(np.arange(len(all_wss))+2,np.array(all_wss),yerr=np.array(std_wss)/np.sqrt(num_repeats))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('total WSS')
    fig.tight_layout()
    # Save the plot to file.
    if plot_file: fig.savefig(plot_file)
    
    return all_wss, std_wss

                    
def wss_over_number_of_combined_clusters(data_a, data_b, label_a = 'Sim A', label_b = 'Sim B', start_frame = 0, 
                                         algorithm='kmeans', max_iter=100, num_repeats = 5, max_num_clusters = 12, 
                                         plot_file = None):
    """
    Calculates the Within-Sum-of-Squares for different numbers of clusters,
    averaged over several iterations.
    
    Parameters
    ----------
        data_a : float array
            Trajectory data [frames,frame_data]
        data_b : float array
            Trajectory data [frames,frame_data]
        label_a : str, optional
            Label for the plot.
        label_b : str, optional
            Label for the plot.
        start_frame : int, optional
            Frame from which the clustering data starts.
        algorithm : string
            The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        max_iter : int, optional
            Maximum number of iterations.
            Default: 100.
        num_repeats : int, optional
            Number of times to run the clustering for each number of clusters.
            Default: 5.
        max_num_clusters : int, optional
            Maximum number of clusters for k-means clustering.
            Default: 12.
        plot_file : str, optional
            Name of the file to save the plot.
        
    Returns
    -------
        all_wss : float array
            WSS values for each number of clusters (starting at 2).
        std_wss : float array
            Standard deviations of the WSS.
    
    """
    
    # Initialize lists
    all_wss = []
    std_wss = []
    # Loop over the number of clusters
    for nc in range(1,max_num_clusters):
        rep_wss = []
        # Run each clustering several times.
        for repeat in range(num_repeats):
            # Get clusters and WSS for this repetition.
            cc = obtain_combined_clusters(data_a, data_b, label_a, label_b, start_frame, 
                                          algorithm=algorithm, max_iter=max_iter, num_clusters=nc, 
                                          plot=False)
            cidx, cond, oidx, wss, centroids = cc
            rep_wss.append(wss)
        # Calculate mean and standard deviation for this number of clusters.
        all_wss.append(np.mean(rep_wss))
        std_wss.append(np.std(rep_wss))
        
    # Plot the WSS over the number of clusters
    fig, ax = plt.subplots(1,1, figsize=[4,3], dpi=300)
    ax.errorbar(np.arange(len(all_wss))+2,np.array(all_wss),yerr=np.array(std_wss)/np.sqrt(num_repeats))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('total WSS')
    fig.tight_layout()
    # Save the plot to file.
    if plot_file: fig.savefig(plot_file)
    
    return all_wss, std_wss
