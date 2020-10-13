import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR CLUSTERING ---
    
    
def obtain_clusters(data,algorithm='kmeans',num_clusters=2,min_dist=12,max_iter=100,
                    plot=True,saveas=None):
    '''Performs a PyEMMA clustering on the provided data
    
    Parameters
    ----------
    data: float array.
        Trajectory data [frames,frame_data]
    algorithm: string
        The algorithm to use for the clustering. 
        Options: kmeans, rspace. Default: kmeans
    num_clusters: int.
        Number of clusters for k-means clustering
    min_dist: float.
        Minimum distance for regspace clustering
    max_iter (opt.): int.
        Maximum number of iterations.
    '''
    
    # Perform PyEMMA clustering
    assert algorithm in ['kmeans','rspace']
    if algorithm == 'kmeans':
        clusters = pyemma.coordinates.cluster_kmeans(data,num_clusters,max_iter=max_iter)
    elif algorithm == 'rspace':
        clusters = pyemma.coordinates.cluster_regspace(data,min_dist)
    
    # Extract cluster indices
    cl_idx = clusters.get_output()[0][:,0]
    
    # Count and plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=[4,3],dpi=300)
        c, nc = np.unique(cl_idx,return_counts=True)
        ax.bar(c,nc)
        if saveas is not None:
            fig.savefig(saveas,dpi=300)
    
    return cl_idx


def obtain_combined_clusters(data_g, data_a, start_frame, label_g, label_a,
                             algorithm='kmeans',num_clusters=2,min_dist=12,max_iter=100,plot=True,saveas=None):
    '''Performs a PyEMMA clustering on a combination of two data sets.
    
    Parameters
    ----------
    data_g: float array.
        Trajectory data [frames,frame_data]
    data_a: float array.
        Trajectory data [frames,frame_data]
    start_frame: int.
        Frame from which the clustering data starts.
    label_g: str.
        Label for the plot.
    label_a: str.
        Label for the plot.
    algorithm: string
        The algorithm to use for the clustering. 
        Options: kmeans, rspace. Default: kmeans
    num_clusters: int.
        Number of clusters for k-means clustering
    min_dist: float.
        Minimum distance for regspace clustering
    max_iter (opt.): int.
        Maximum number of iterations.
    saveas (opt.): str.
        Name of the file to save the plot
    '''
    
    # Combine the data
    data = np.concatenate([data_g,data_a],0)

    # Remember which simulation the data came frome
    cond = np.concatenate([np.ones(len(data_g)), np.zeros(len(data_a))])

    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(len(data_g))+start_frame, np.arange(len(data_a))+start_frame])

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
        cg, ncg = np.unique(cidx[cond==1],return_counts=True)
        ca, nca = np.unique(cidx[cond==0],return_counts=True)
        ax.bar(cg-0.15,ncg,0.3,label=label_g)
        ax.bar(ca+0.15,nca,0.3,label=label_a)
        ax.legend()
        ax.set_xticks(c)
        ax.set_xlabel('clusters')
        ax.set_ylabel('population')
        fig.tight_layout()
        if saveas is not None:
            fig.savefig(saveas,dpi=300)
    
    return cidx, cond, oidx, total_wss, centroids


def write_cluster_traj(cluster_idx, top_file, trj_file, out_name, start_frame):
    '''Writes a trajectory into a separate file for each cluster.'''
    
    u = mda.Universe(top_file, trj_file)
    protein = u.select_atoms('all')
    
    num_clusters = np.max(cluster_idx)+1
    
    for nr in range(num_clusters):
        with mda.Writer(out_name+"_c"+str(nr)+".xtc", protein.n_atoms) as W:
            for ts in u.trajectory:
                if ts.frame >= start_frame and cluster_idx[ts.frame-start_frame] == nr: 
                    W.write(protein)
                    
                    
def wss_over_number_of_clusters(data_a, data_b, label_a = 'Sim A', label_b = 'Sim B', start_frame = 0, 
                                algorithm='kmeans', max_iter=100, num_repeats = 5, max_num_clusters = 12, 
                                plot_file = None):
    '''Calculates the Within-Sum-of-Squares for different numbers of clusters,
       averaged over several iterations.'''
    
    all_wss = []
    std_wss = []
    for nc in range(1,max_num_clusters):
        rep_wss = []
        for repeat in range(num_repeats):
            cc = obtain_combined_clusters(data_a, data_b, start_frame, 
                                          label_a, label_b,  
                                          algorithm=algorithm, max_iter=max_iter, 
                                          num_clusters=nc, plot=False)
            cidx, cond, oidx, wss, centroids = cc
            rep_wss.append(wss)

        all_wss.append(np.mean(rep_wss))
        std_wss.append(np.std(rep_wss))
        
    fig, ax = plt.subplots(1,1, figsize=[4,3], dpi=300)
    ax.errorbar(np.arange(len(all_wss))+2,np.array(all_wss),yerr=np.array(std_wss)/np.sqrt(num_repeats))
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('total WSS')
    fig.tight_layout()
    if plot_file: fig.savefig(plot_file)
    
    return all_wss, std_wss
    
    

                    
