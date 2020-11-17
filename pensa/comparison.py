import numpy as np
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR FEATURE DIFFERENCE ANALYSIS ---


def relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=0.001, verbose=True):
    """
    Calculates the Jensen-Shannon distance and the Kullback-Leibler divergences for each feature from two ensembles.
    
    Args:
        features_a (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
        features_b (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check. 
        all_data_a (float array): Trajectory data from the first ensemble [frames,frame_data].
        all_data_b (float array): Trajectory data from the second ensemble [frames,frame_data].
        bin_width (float): Bin width for the axis to compare the distributions on. Defaults to 0.001.
        verbose (bool): Print intermediate results. Defaults to True.
    
    Returns:
        data_names (list of str): Feature names.
        data_jsdist (float array): Jensen-Shannon distance for each feature.
        data_kld_ab (float array): Kullback-Leibler divergences of data_a wrt to data_b.
        data_kld_ba (float array): Kullback-Leibler divergences of data_b wrt to data_a.
        
    """
    
    all_data_a, all_data_b = all_data_a.T, all_data_b.T
    
    # Assert that the features are the same and data sets have same number of features
    assert features_a.describe() == features_b.describe()
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    
    # Extract the names of the features
    data_names = features_a.active_features[0].describe()
    
    # Initialize relative entropy and average value
    data_jsdist = np.zeros(len(data_names))
    data_kld_ab = np.zeros(len(data_names))
    data_kld_ba = np.zeros(len(data_names))
    data_avg    = np.zeros(len(data_names))
    
    # Loop over all features
    for i in range(len(all_data_a)):
        
        data_a = all_data_a[i]
        data_b = all_data_b[i]
 
        # Combine both data sets
        data_both = np.concatenate((data_a,data_b))
        data_avg[i] = np.mean(data_both)
        
        # Get bin values for all histograms from the combined data set
        bins_min = np.min( data_both )
        bins_max = np.max( data_both )
        bins = np.arange(bins_min,bins_max,bin_width)

        # Calculate histograms for combined and single data sets
        histo_both = np.histogram(data_both, density = True)
        histo_a = np.histogram(data_a, density = True, bins = histo_both[1])
        distr_a = histo_a[0] / np.sum(histo_a[0])
        histo_b = np.histogram(data_b, density = True, bins = histo_both[1])
        distr_b = histo_b[0] / np.sum(histo_b[0])
        
        # Calculate relative entropies between the two data sets (Kullback-Leibler divergence)
        data_kld_ab[i] = np.sum( sp.special.kl_div(distr_a,distr_b) )
        data_kld_ba[i] = np.sum( sp.special.kl_div(distr_b,distr_a) )
        
        # Calculate the Jensen-Shannon distance
        data_jsdist[i] = scipy.spatial.distance.jensenshannon(distr_a, distr_b, base=2.0)
        
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f %1.2f"%(js_dist,rel_ent_ab,rel_ent_ba))
        
    return data_names, data_jsdist, data_kld_ab, data_kld_ba



def kolmogorov_smirnov_analysis(features_a, features_g, all_data_a, all_data_g, bin_width=0.001, verbose=True):
    """
    Calculates Kolmogorov-Smirnov statistic for two distributions.
    
    Args:
        features_a (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
        features_b (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check. 
        all_data_a (float array): Trajectory data from the first ensemble [frames,frame_data].
        all_data_b (float array): Trajectory data from the second ensemble [frames,frame_data].
        bin_width (float): Bin width for the axis to compare the distributions on. Defaults to 0.001.
        verbose (bool): Print intermediate results. Defaults to True.

    Returns:
        data_names (list of str): Feature names.
        data_kss (float array): Kolmogorov-Smirnov statistics for each feature.
        data_ksp (float array): Kolmogorov-Smirnov p-value for each feature.
        
    """

    all_data_a, all_data_g = all_data_a.T, all_data_g.T
    
    # Assert that features are the same and data sets have same number of features
    assert features_a.describe() == features_g.describe()
    assert all_data_a.shape[0] == all_data_g.shape[0] 
    
    # Extract names of features
    data_names = features_a.active_features[0].describe()

    # Initialize relative entropy and average value
    data_avg = np.zeros(len(data_names))
    data_kss = np.zeros(len(data_names))
    data_ksp = np.zeros(len(data_names))

    for i in range(len(all_data_a)):

        data_a = all_data_a[i]
        data_g = all_data_g[i]
        
        # Perform Kolmogorov-Smirnov test
        ks = sp.stats.ks_2samp(data_a,data_g)
        data_kss[i] = ks.statistic
        data_ksp[i] = ks.pvalue
        
        # Combine both data sets
        data_both = np.concatenate((data_a,data_g))
        data_avg[i] = np.mean(data_both)
        
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f"%(ks.statistic,ks.pvalue) )
        
    return data_names, data_kss, data_ksp



def mean_difference_analysis(features_a, features_g, all_data_a, all_data_g, verbose=True):
    """
    Compares the arithmetic means of two distance distributions.
    
    Args:
        features_a (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
        features_b (list of str): Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check. 
        all_data_a (float array): Trajectory data from the first ensemble [frames,frame_data].
        all_data_b (float array): Trajectory data from the second ensemble [frames,frame_data].
        bin_width (float): Bin width for the axis to compare the distributions on. Defaults to 0.001.
        verbose (bool): Print intermediate results. Defaults to True.

    Returns:
        data_names (list of str): Feature names.
        data_avg (float array): Joint average value for each feature.
        data_diff (float array): Difference of the averages for each feature.
        
    """
    
    all_data_a, all_data_g = all_data_a.T, all_data_g.T
    
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



def get_feature_timeseries(feat, data, feature_type, feature_name):
    """
    Returns the timeseries of one particular feature.
    
    Args:
        feat (features obj): Object with all feature names.
        data (float array): Trajectory data from the simulation.
        feature_type: Type of the selected feature 
            ('bb-torsions', 'bb-distances', 'sc-torsions').
        feature_name: Name of the selected feature.
    
    Returns:
        timeseries (float array): Value of the feature for each frame.
    
    """
    # Select the feature and get its index.
    index = np.where( np.array( feat[feature_type].describe() ) == feature_name )[0][0]
    # Extract the timeseries.
    timeseries = data[feature_type][:,index]
    return timeseries



def sort_features(names, sortby):
    """
    Sorts features by a list of values.
    
    Args:
        names (str array): Array of feature names.
        sortby (float array): Array of the values to sort the names by.
        
    Returns:
        sort (array of tuples [str,float]): Array of sorted tuples with feature and value.
    
    """
    # Get the indices of the sorted order
    sort_id = np.argsort(sortby)[::-1]  
    # Bring the names and values in the right order
    sorted_names  = []
    sorted_values = []
    for i in sort_id:
        sorted_names.append(np.array(names)[i])
        sorted_values.append(sortby[i])
    sn, sv = np.array(sorted_names), np.array(sorted_values)
    # Format for output
    sort = np.array([sn,sv]).T
    return sort



def residue_visualization(names, data, ref_filename, pdf_filename, pdb_filename, 
                          selection='max', y_label='max. JS dist. of BB torsions'):
    """
    Visualizes features per residue as plot and in PDB files, assuming values from 0 to 1. 
    
    Args:
        names (str array): Names of the features in PyEMMA nomenclaturre (contain residue ID).
        data (float array): Data to project onto the structure.
        ref_filename (str): Name of the file for the reference structure.
        pdf_filename (str): Name of the PDF file to save the plot.
        pdb_filename (str): Name of the PDB file to save the structure with the values to visualize.
        selection (str, optional): How to select the value to visualize for each residue from all its features 
            Options: 'max', 'min'. Default: 'max'
        y_label (str, optional): Label of the y axis of the plot.
            Default: 'max. JS dist. of BB torsions'
        
    Returns:
        vis_resids (int array): Residue numbers.
        vis_values (float array): Values of the quantity to be visualized.
         
    """
    
    # -- INITIALIZATION --
    
    # Structure to use for visualization
    u = mda.Universe(ref_filename)
    vis_resids = u.residues.resids
    # Output values
    default = 0 if selection=='max' else 1
    vis_values = default*np.ones(len(vis_resids))
    
    # -- VALUE ASSIGNMENT --

    for i,name in enumerate(names):
        # To each residue ...
        resid = int( name.split(' ')[-1][:-1] )
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
                            vmin=None, vmax=None, verbose=True):
    """
    Visualizes features per residue as plot and in PDB files, assuming values from 0 to 1. 
    
    Args:
        dist_names (str array): Names of the features in PyEMMA nomenclaturre (contain residue IDs).
        dist_diff (float array): Data for each distance feature.
        plot_filename (str): Name of the file for the plot.
        vmin (float, optional): Minimum value for the heat map.
        vmax (float, optional): Maximum value for the heat map.
        verbose (bool, optional): Print numbers of first and last residue. Defaults to True.
        
    Returns:
        diff (float array): Distance matrix.
         
    """
    
    # Calculate the distance Matrix
    firstres = int(dist_names[0].split(' ')[2])
    lastres  = int(dist_names[-1].split(' ')[2])
    if verbose:
        print('Plotting distance matrix')
        print('first res:', firstres, ', last res:', lastres)
    size = lastres-firstres+2
    diff = np.zeros([size,size])
    for n,name in enumerate(dist_names):
        splitname = name.split(' ')
        resi,resj = int(splitname[2]),int(splitname[7])
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
    fig.colorbar(img,ax=ax)
    fig.tight_layout()
    fig.savefig(plot_filename,dpi=300)  
    
    return diff


