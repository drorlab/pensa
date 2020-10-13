import numpy as np
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt



# --- METHODS FOR FEATURE DIFFERENCE ANALYSIS ---


def relative_entropy_analysis(features_a, features_g, all_data_a, all_data_g, bin_width=0.001, verbose=True):
    """
    Calculates Jensen-Shannon distance and Kullback-Leibler divergences for two distributions.
    """
    
    all_data_a, all_data_g = all_data_a.T, all_data_g.T
    
    # Assert that features are the same and data sets have same number of features
    assert features_a.describe() == features_g.describe()
    assert all_data_a.shape[0] == all_data_g.shape[0] 
    
    # Extract names of features
    data_names = features_a.active_features[0].describe()

    # Initialize relative entropy and average value
    data_jsdist = np.zeros(len(data_names))
    data_kld_ag = np.zeros(len(data_names))
    data_kld_ga = np.zeros(len(data_names))
    data_avg    = np.zeros(len(data_names))

    for i in range(len(all_data_a)):

        data_a = all_data_a[i]
        data_g = all_data_g[i]
        
        # Combine both data sets
        data_both = np.concatenate((data_a,data_g))
        data_avg[i] = np.mean(data_both)
        
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
        data_kld_ag[i] = np.sum( sp.special.kl_div(distr_a,distr_g) )
        data_kld_ga[i] = np.sum( sp.special.kl_div(distr_g,distr_a) )
        
        # Calculate the Jensen-Shannon distance
        data_jsdist[i] = scipy.spatial.distance.jensenshannon(distr_a, distr_g, base=2.0)
        
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f %1.2f"%(js_dist,rel_ent_ag,rel_ent_ga))
        
    return data_names, data_jsdist, data_kld_ag, data_kld_ga



def kolmogorov_smirnov_analysis(features_a, features_g, all_data_a, all_data_g, bin_width=0.001, verbose=True):
    """
    Calculates Kolmogorov-Smirnov statistic for two distributions.
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
    
    index = np.where( np.array( feat[feature_type].describe() ) == feature_name )[0][0]

    timeseries = data[feature_type][:,index]
    
    return timeseries



def sort_features(names, sortby):
    """
    Sort features by a list of values.
    
    Parameters
    ----------
    names: str array
        array of feature names.
    sortby: float array
        array of the values to sort the names by.
        
    Returns
    -------
    sort: array of tuples (str, float)
        array of sorted tuples with feature and value
    
    """

    sort_id = np.argsort(sortby)[::-1]  
    
    sorted_names = []
    sorted_values = []
    
    for i in sort_id:
        sorted_names.append(np.array(names)[i])
        sorted_values.append(sortby[i])
        
    sn, sv = np.array(sorted_names), np.array(sorted_values)
    
    sort = np.array([sn,sv]).T
    
    return sort



def residue_visualization(names, data, ref_filename, pdf_filename, pdb_filename, 
                          selection='max', y_label='max. JS dist. of BB torsions'):
    """
    Visualizes features per residue as plot and in PDB files, assuming values from 0 to 1. 
    
    Parameters
    ----------
    names: str array
        Names of the features in PyEMMA nomenclaturre (contain residue ID).
    data: float array
        Data to project onto the structure.
    ref_filename: str
        Name of the file for the reference structure.
    pdf_filename: str
        Name of the PDF file to save the plot.
    pdb_filename: str
        Name of the PDB file to save the structure with the values to visualize.
    selection: str ['max', 'min']
        How to select the value to visualize for each residue from all its features.
    y_label: str
        Label of the y axis of the plot.
        
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
    ax.plot(vis_resids, vis_values, '.')
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



def sort_distances(dist_names, dist_avg, dist_relen, dist_id):

    # Define the criteria
    min_rel_ent = 0.2
    min_av_dist = 0.0 # [nm]
    max_av_dist = 2.5 # [nm] 
    min_id_diff = 300 # minimum difference of the atom numbers 

    # Extract IDs of the atoms used
    dist_id_diff = dist_id[:,1] - dist_id[:,0]

    # Combine the criteria
    criteria =  (dist_id_diff > min_id_diff) 
    criteria *= (dist_relen > min_rel_ent)
    criteria *= (dist_avg < max_av_dist)
    criteria *= (dist_avg > min_av_dist)

    # Sort the distances not filtered out by the criteria
    sort_id = np.argsort(dist_relen[criteria])[::-1]

    # ... and print them
    for i in sort_id:
        print(np.array(dist_names)[criteria][i], 
              '; %1.3f'%dist_avg[criteria][i], 
              '; %1.3f'%dist_relen[criteria][i] )
        
        

def distances_visualization(dist_names, dist_diff, out_filename, 
                            vmin=None, vmax=None, verbose=False):
        
    # Distance Matrix
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
            
    # Plot
    fig,ax = plt.subplots(1,1,figsize=[6,4],dpi=300)
    img = ax.imshow(diff, vmin=vmin, vmax=vmax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(50-firstres,lastres-firstres+1,50))
    ax.set_yticks(np.arange(50-firstres,lastres-firstres+1,50))
    ax.set_xticklabels(np.arange(50,lastres+1,50))
    ax.set_yticklabels(np.arange(50,lastres+1,50))
    fig.colorbar(img,ax=ax)
    fig.tight_layout()
    fig.savefig(out_filename,dpi=300)  
    
    return diff


