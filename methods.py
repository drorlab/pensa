import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



def get_features(pdb,xtc,start_frame):
    
    labels = []
    features = []
    data = []
    
    torsions_feat  = pyemma.coordinates.featurizer(pdb)
    torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    torsions_data = pyemma.coordinates.load(xtc, features=torsions_feat)[start_frame:]
    labels   = ['backbone\ntorsions']
    features = [torsions_feat]
    data     = [torsions_data]
    
    positions_feat = pyemma.coordinates.featurizer(pdb)
    positions_feat.add_selection(positions_feat.select_Backbone())
    positions_data = pyemma.coordinates.load(xtc, features=positions_feat)[start_frame:]
    labels   += ['backbone atom\npositions']
    features += [positions_feat]
    data     += [positions_data]
    
    distances_feat = pyemma.coordinates.featurizer(pdb)
    distances_feat.add_distances(distances_feat.pairs(distances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
    distances_data = pyemma.coordinates.load(xtc, features=distances_feat)[start_frame:]
    labels   += ['backbone atom\ndistances']
    features += [distances_feat]
    data     += [distances_data]
    
    sidechains_feat  = pyemma.coordinates.featurizer(pdb)
    sidechains_feat.add_sidechain_torsions(cossin=True, periodic=False)
    sidechains_data = pyemma.coordinates.load(xtc, features=sidechains_feat)[start_frame:]
    labels   += ['sidechains\ntorsions']
    features += [sidechains_feat]
    data     += [sidechains_data]
    
    return labels, features, data



# --- METHODS FOR FEATURE DIFFERENCE ANALYSIS ---


def relative_entropy_analysis(features_a, features_g, all_data_a, all_data_g, bin_width=0.001, verbose=True):
    """Calculates Jensen-Shannon distance, Kullback-Leibler divergences, and Kolmogorov-Smirnov statistic for the two distributions."""
    
    # Assert that features are the same and data sets have same number of features
    assert features_a.describe() == features_g.describe()
    assert all_data_a.shape[0] == all_data_g.shape[0] 
    
    # Extract names of features
    data_names = features_a.active_features[0].describe()

    # Initialize relative entropy and average value
    data_relen = np.zeros([len(data_names),3])
    data_avg   = np.zeros(len(data_names))
    data_ks    = np.zeros([len(data_names),2])

    for i in range(len(all_data_a)):

        data_a = all_data_a[i]
        data_g = all_data_g[i]
        
        # Perform Kolmogorov-Smirnov test
        ks = sp.stats.ks_2samp(data_a,data_g)
        data_ks[i] = np.array([ks.statistic,ks.pvalue])
        
        # Combine both data sets
        data_both = np.concatenate((data_a,data_g))

        # Get bin values for all histograms from the combined data set
        bins_min = np.min( data_both )
        bins_max = np.max( data_both )
        bins = np.arange(bins_min,bins_max,bin_width)

        # Calculate histograms for combined and single data sets
        histo_both = np.histogram(data_both, density = True)
        histo_a = np.histogram(data_a, density = True, bins = histo_both[1])
        distr_a = histo_a[0] / np.sum(histo_a[0])
        histo_g = np.histogram(data_g, density = True, bins = histo_both[1])
        distr_g = histo_g[0] / np.sum(histo_g[0])
        
        # Calculate relative entropies between the two data sets (Kullback-Leibler divergence)
        rel_ent_ag = np.sum( sp.special.kl_div(distr_a,distr_g) )
        rel_ent_ga = np.sum( sp.special.kl_div(distr_g,distr_a) )
        
        # Calculate the Jensen-Shannon distance
        js_dist = scipy.spatial.distance.jensenshannon(distr_a, distr_g, base=2.0)
        
        # Update the output arrays
        data_avg[i] = np.mean(data_both)
        data_relen[i] = np.array([js_dist,rel_ent_ag,rel_ent_ga])
        
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f %1.2f"%(js_dist,rel_ent_ag,rel_ent_ga), 
                  " %1.2f %1.2f"%(ks.statistic,ks.pvalue) )
        
    return data_names, data_avg, data_relen, data_ks



def mean_difference_analysis(features_a, features_g, all_data_a, all_data_g, verbose=True):
    """Compares the arithmetic means of two distance distributions."""
    
    # Assert that features are the same and data sets have same number of features
    assert features_a.describe() == features_g.describe()
    assert all_data_a.shape[0] == all_data_g.shape[0] 
    
    # Extract names of features
    data_names = features_a.active_features[0].describe()

    # Initialize relative entropy and average value
    data_diff = np.zeros(len(data_names))
    data_avg  = np.zeros(len(data_names))

    for i in range(len(all_data_a)):

        data_a = all_data_a[i]
        data_g = all_data_g[i]

        # Calculate means of the data sets
        mean_a = np.mean(data_a)
        mean_g = np.mean(data_g)

        # Calculate difference of means between the two data sets
        diff_ag = mean_a-mean_g
        mean_ag = 0.5*(mean_a+mean_g)

        # Update the output arrays
        data_avg[i]  = mean_ag
        data_diff[i] = diff_ag
        
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f"%data_diff[i])
        
    return data_names, data_avg, data_diff



# --- METHODS FOR PRINCIPAL COMPONENT ANALYSIS ---


def calculate_pca(data):
    '''Performs a PyEMMA PCA on the provided data
    
    Parameters
    ----------
    data: float array.
        Trajectory data [frames,frame_data]
    '''
    
    pca = pyemma.coordinates.pca(data)
    
    # Plot the eigenvalues
    fig,ax = plt.subplots(1,1,figsize=[4,3],dpi=100)
    ax.plot(pca.eigenvalues[:12],'o')
    plt.show()   
    
    return pca


def pca_features(pca,features,num,threshold):
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
    
    
def project_on_pc(data,ev_idx,pca=None):
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
    
    if pca is None:
        pca = pyemma.coordinates.pca(data) #,dim=3)

    projection = np.zeros(data.shape[0])
    for ti in range(data.shape[0]):
        projection[ti] = np.dot(data[ti],pca.eigenvectors[:,ev_idx])
        
    return projection
    

def sort_trajs_along_common_pc(data_g,data_a,start_frame,name_g,name_a,sim):
    '''Sort two trajectories along the 12 highest principal components.
    
    Parameters
    ----------
    data_g: float array.
        Trajectory data [frames,frame_data]
    data_a: float array.
        Trajectory data [frames,frame_data]
    start_frame: int
        offset of the data with respect to the trajectories (defined below)
    name_g: string.
        first of the trajetories from which the frames are picked (g-bound). 
        Should be the same as data_g was from.
    name_a: string.
        first of the trajetories from which the frames are picked (arr-bound). 
        Should be the same as data_g was from.
    sim: string.
        simulation type (tremd or mremd)
    '''
    
    # Combine the data
    data = np.concatenate([data_g,data_a],0)
    
    # Remember which simulation the data came frome
    cond = np.concatenate([np.ones(len(data_g)), np.zeros(len(data_a))])

    # Remember the index in the respective simulation (taking into account cutoff)
    oidx = np.concatenate([np.arange(len(data_g))+start_frame, 
                           np.arange(len(data_a))+start_frame])
    
    # Calculate the principal components
    pca = pyemma.coordinates.pca(data,dim=3)

    # Define the MDAnalysis trajectories from where the frames come
    ug = mda.Universe("traj/"+name_g+".gro","traj/"+name_g+"_"+sim+".xtc")
    ua = mda.Universe("traj/"+name_a+".gro","traj/"+name_a+"_"+sim+".xtc")

    ag = ug.select_atoms('all')
    aa = ua.select_atoms('all')

    for evi in range(12):

        # Project the combined data on the principal component
        proj = project_on_pc(data,evi,pca=pca)

        # Sort everything along the projection on th resp. PC
        sort_idx  = np.argsort(proj)
        proj_sort = proj[sort_idx] 
        cond_sort = cond[sort_idx]
        oidx_sort = oidx[sort_idx]

        with mda.Writer("pca/"+name_g.split('_')[0]+"_"+name_g.split('_')[-1]+"_"+sim+"_pc"+str(evi)+".xtc", ag.n_atoms) as W:
            for i in range(data.shape[0]):
                if cond_sort[i] == 1: # G-protein bound
                    ts = ug.trajectory[oidx_sort[i]]
                    W.write(ag)
                elif cond_sort[i] == 0: # arrestin bound
                    ts = ua.trajectory[oidx_sort[i]]
                    W.write(aa)
                    
     
    
# --- METHODS FOR CLUSTERING ---
    
    
def obtain_clusters(data,algorithm='kmeans',num_clusters=2,min_dist=12,max_iter=100):
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
    fig,ax = plt.subplots(1,1,figsize=[4,3])
    c, nc = np.unique(cl_idx,return_counts=True)
    ax.bar(c,nc)
    
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
        cluster_wss = np.sum( (cluster_data - centroid)**2 )
        total_wss += cluster_wss
    
    # Count and plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=[4,3],sharex=True,dpi=100)
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


def write_cluster_traj(cluster_idx,base_name,name,sim,start_frame):
    '''Writes a trajectory into a separate file for each cluster.'''
    
    print("traj/"+name+".gro","traj/"+name+"_"+sim+".xtc")
    u = mda.Universe("traj/"+name+".gro","traj/"+name+"_"+sim+".xtc")
    protein = u.select_atoms('all')
    
    num_clusters = np.max(cluster_idx)+1
    
    for nr in range(num_clusters):
        with mda.Writer("clusters/"+base_name+"_"+name+"_"+sim+"_c"+str(nr)+".xtc", protein.n_atoms) as W:
            for ts in u.trajectory:
                if ts.frame >= start_frame and cluster_idx[ts.frame-start_frame] == nr: 
                    W.write(protein)
                    
                    