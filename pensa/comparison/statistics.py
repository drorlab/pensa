import numpy as np
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
from pensa.features import *



def kolmogorov_smirnov_analysis(features_a, features_b, all_data_a, all_data_b, verbose=True,
                                override_name_check=False):
    """
    Calculates Kolmogorov-Smirnov statistic for two distributions.
    
    Parameters
    ----------
        features_a : list of str
            Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
        features_b : list of str
            Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check. 
        all_data_a : float array
            Trajectory data from the first ensemble. Format: [frames,frame_data].
        all_data_b : float array
            Trajectory data from the second ensemble. Format: [frames,frame_data].
        verbose : bool, default=True
            Print intermediate results.
        override_name_check : bool, default=False
            Only check number of features, not their names.

    Returns
    -------
        data_names : list of str
            Feature names.
        data_kss : float array
            Kolmogorov-Smirnov statistics for each feature.
        data_ksp : float array
            Kolmogorov-Smirnov p-value for each feature.
        
    """
    all_data_a, all_data_b = all_data_a.T, all_data_b.T
    # Assert that features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract names of features
    data_names = features_a
    # Initialize relative entropy and average value
    data_avg = np.zeros(len(data_names))
    data_kss = np.zeros(len(data_names))
    data_ksp = np.zeros(len(data_names))
    # Loop over all features
    for i in range(len(all_data_a)):
        data_a = all_data_a[i]
        data_b = all_data_b[i]
        # Perform Kolmogorov-Smirnov test
        ks = sp.stats.ks_2samp(data_a,data_b)
        data_kss[i] = ks.statistic
        data_ksp[i] = ks.pvalue        
        # Combine both data sets
        data_both = np.concatenate((data_a,data_b))
        data_avg[i] = np.mean(data_both)
        # Print information
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f"%(ks.statistic,ks.pvalue) )        
    return data_names, data_kss, data_ksp



def mean_difference_analysis(features_a, features_b, all_data_a, all_data_b, verbose=True, 
                             override_name_check=False):
    """
    Compares the arithmetic means of two distance distributions.
    
    Parameters
    ----------
        features_a : list of str
            Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
        features_b : list of str
            Feature names of the first ensemble. 
            Can be obtained from features object via .describe().
            Must be the same as features_a. Provided as a sanity check. 
        all_data_a : float array
            Trajectory data from the first ensemble. Format: [frames,frame_data].
        all_data_b : float array
            Trajectory data from the second ensemble. Format: [frames,frame_data].
        bin_width : float, default=0.001
            Bin width for the axis to compare the distributions on.
        verbose : bool, default=True
            Print intermediate results.
        override_name_check : bool, default=False
            Only check number of features, not their names.

    Returns
    -------
        data_names : list of str
            Feature names.
        data_avg : float array
            Joint average value for each feature.
        data_diff : float array
            Difference of the averages for each feature.
        
    """
    all_data_a, all_data_b = all_data_a.T, all_data_b.T
    # Assert that features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract names of features
    data_names = features_a
    # Initialize relative entropy and average value
    data_diff = np.zeros(len(data_names))
    data_avg  = np.zeros(len(data_names))
    # Loop over all features
    for i in range(len(all_data_a)):
        data_a = all_data_a[i]
        data_b = all_data_b[i]
        # Calculate means of the data sets
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        # Calculate difference of means between the two data sets
        diff_ab = mean_a-mean_b
        mean_ab = 0.5*(mean_a+mean_b)
        # Update the output arrays
        data_avg[i]  = mean_ab
        data_diff[i] = diff_ab
        # Print information
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f"%data_diff[i])    
    return data_names, data_avg, data_diff





