import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR CLUSTERING ---
    
    
def obtain_clusters(data, algorithm='kmeans', 
                    num_clusters=2, min_dist=12, max_iter=100,
                    plot=True, saveas=None):
    """
    Clusters the provided data.
    
    Args:
        data (float array): Trajectory data [frames,frame_data]
        algorithm (string): The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        num_clusters (int, optional): Number of clusters for k-means clustering. 
            Default: 2.
        min_dist (float, optional): Minimum distance for regspace clustering. 
            Default: 12.
        max_iter (int, optional): Maximum number of iterations. 
            Default: 100.
        plot (bool, optional): Create a plot. 
            Default: True
        saveas (str, optional): Name of the file in which to save the plot.
            (only needed if "plot" is True)
        
    Returns:
        cidx (int array): Cluster indices for each frame.
        total_wss (float): With-in-sum-of-squares (WSS).
        centroids (float array): Centroids for all the clusters.
    
    """
    
    # Perform PyEMMA clustering
    assert algorithm in ['kmeans','rspace']
    if algorithm == 'kmeans':
        clusters = pyemma.coordinates.cluster_kmeans(data,num_clusters,max_iter=max_iter)
    elif algorithm == 'rspace':
        clusters = pyemma.coordinates.cluster_regspace(data,min_dist)
    
    # Extract cluster indices
    cidx = clusters.get_output()[0][:,0]
    
    # Calculate centroids and total within-cluster sum of square
    centroids = []
    total_wss = 0
    for i in np.unique(cidx):
        # get the data for this cluster
        cluster_data = data[np.where(cidx==i)]
        # calcualte the centroid
        cluster_centroid = np.mean(cluster_data,0)
        centroids.append(cluster_centroid)
        # calculate the within-cluster sum of square
        cluster_wss = np.sum( (cluster_data - cluster_centroid)**2 )
        total_wss += cluster_wss
        
    # Count and plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=[4,3],dpi=300)
        c, nc = np.unique(cidx,return_counts=True)
        ax.bar(c,nc)
        if saveas is not None:
            fig.savefig(saveas,dpi=300)
    
    return cidx, total_wss, centroids


def obtain_combined_clusters(data_a, data_b, label_a = 'Sim A', label_b = 'Sim B', start_frame = 0,
                             algorithm='kmeans', num_clusters=2, min_dist=12, max_iter=100,
                             plot=True, saveas=None):
    """
    Clusters a combination of two data sets.
    
    Args:
        data_a (float array): Trajectory data [frames,frame_data]
        data_b (float array): Trajectory data [frames,frame_data]
        label_a (str, optional): Label for the plot.
            Default: Sim A.
        label_b (str, optional): Label for the plot.
            Default: Sim B.
        start_frame (int): Frame from which the clustering data starts.
            Default: 0.
        algorithm (string): The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        num_clusters (int, optional): Number of clusters for k-means clustering. 
            Default: 2.
        min_dist (float, optional): Minimum distance for regspace clustering. 
            Default: 12.
        max_iter (int, optional): Maximum number of iterations. 
            Default: 100.
        plot (bool, optional): Create a plot. 
            Default: True
        saveas (str, optional): Name of the file in which to save the plot.
            (only needed if "plot" is True)
        
    Returns:
        cidx (int array): Cluster indices for each frame.
        cond (int array): Index of the simulation the data came frome.
        oidx (int array): Index of each frame in the original simulation (taking into account cutoff)
        total_wss (float): With-in-sum-of-squares (WSS).
        centroids (float array): Centroids for all the clusters.
           
    """
    
    # Combine the data
    data = np.concatenate([data_a,data_b],0)

    # Remember which simulation the data came frome
    cond = np.concatenate([np.ones(len(data_a)), np.zeros(len(data_b))])

    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(len(data_a))+start_frame, np.arange(len(data_b))+start_frame])

    # Perform PyEMMA clustering
    assert algorithm in ['kmeans','rspace']
    if algorithm == 'kmeans':
        clusters = pyemma.coordinates.cluster_kmeans(data,k=num_clusters,max_iter=100)
    elif algorithm == 'rspace':
        clusters = pyemma.coordinates.cluster_regspace(data,min_dist)

    # Extract cluster indices
    cidx = clusters.get_output()[0][:,0]

    # Calculate centroids and total within-cluster sum of square
    centroids = []
    total_wss = 0
    for i in np.unique(cidx):
        # get the data for this cluster
        cluster_data = data[np.where(cidx==i)]
        # calcualte the centroid
        cluster_centroid = np.mean(cluster_data,0)
        centroids.append(cluster_centroid)
        # calculate the within-cluster sum of square
        cluster_wss = np.sum( (cluster_data - cluster_centroid)**2 )
        total_wss += cluster_wss
    
    # Count and plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=[4,3],sharex=True,dpi=300)
        c, nc   = np.unique(cidx,return_counts=True)
        ca, nca = np.unique(cidx[cond==1],return_counts=True)
        cb, ncb = np.unique(cidx[cond==0],return_counts=True)
        ax.bar(ca-0.15,nca,0.3,label=label_a)
        ax.bar(cb+0.15,ncb,0.3,label=label_b)
        ax.legend()
        ax.set_xticks(c)
        ax.set_xlabel('clusters')
        ax.set_ylabel('population')
        fig.tight_layout()
        if saveas is not None:
            fig.savefig(saveas,dpi=300)
    
    return cidx, cond, oidx, total_wss, centroids


def write_cluster_traj(cluster_idx, top_file, trj_file, out_name, start_frame=0):
    """
    Writes a trajectory into a separate file for each cluster.
    
    Args:
        cluster_idx (int array): Cluster index for each frame.
        top_file (str): Reference topology for the second trajectory. 
        trj_file (str): Trajetory file from which the frames are picked.
        out_name (str): Core part of the name of the output files.
        start_frame (int, optional): Frame from which to start reading the trajectory.
        
    """
    
    # Load and select the protein
    u = mda.Universe(top_file, trj_file)
    protein = u.select_atoms('all')
    
    # Loop over clusters
    num_clusters = np.max(cluster_idx)+1
    for nr in range(num_clusters):
        # For each cluster, write the corresponding frames to their new trajectory.
        with mda.Writer(out_name+"_c"+str(nr)+".xtc", protein.n_atoms) as W:
            for ts in u.trajectory:
                if ts.frame >= start_frame and cluster_idx[ts.frame-start_frame] == nr: 
                    W.write(protein)
    return
                    

def wss_over_number_of_clusters(data, algorithm='kmeans', 
                                max_iter=100, num_repeats = 5, max_num_clusters = 12,
                                plot_file = None):
    """
    Calculates the within-sum-of-squares (WSS) for different numbers of clusters,
    averaged over several iterations.
    
    Args:
        data (float array): Trajectory data [frames,frame_data]
        algorithm (string): The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        max_iter (int, optional): Maximum number of iterations. 
            Default: 100.
        num_repeats (int, optional): Number of times to run the clustering for each number of clusters.
            Default: 5.
        max_num_clusters (int, optional): Maximum number of clusters for k-means clustering. 
            Default: 12.
        plot_file (str, optional): Name of the file to save the plot.
        
    Returns:
        all_wss (float array): WSS values for each number of clusters (starting at 2).
        std_wss (float array): standard deviations of the WSS.
        
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
    
    Args:
        data_a (float array): Trajectory data [frames,frame_data]
        data_b (float array): Trajectory data [frames,frame_data]
        label_a (str, optional): Label for the plot.
        label_b (str, optional): Label for the plot.
        start_frame (int, optional): Frame from which the clustering data starts.
        algorithm (string): The algorithm to use for the clustering. 
            Options: kmeans, rspace. 
            Default: kmeans
        max_iter (int, optional): Maximum number of iterations.
            Default: 100.
        num_repeats (int, optional): Number of times to run the clustering for each number of clusters.
            Default: 5.
        max_num_clusters (int, optional): Maximum number of clusters for k-means clustering.
            Default: 12.
        plot_file (str, optional): Name of the file to save the plot.
        
    Returns:
        all_wss (float array): WSS values for each number of clusters (starting at 2).
        std_wss (float array): standard deviations of the WSS.
    
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

