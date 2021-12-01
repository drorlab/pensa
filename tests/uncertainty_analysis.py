import numpy as np
from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *
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

# -- Functions to calculate SSI statistics across paired ensembles --


def unc_relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, verbose=True, override_name_check=False, block_length=None, cumdist=True):
    """
    

    Parameters
    ----------
    features_a : TYPE
        DESCRIPTION.
    features_b : TYPE
        DESCRIPTION.
    all_data_a : TYPE
        DESCRIPTION.
    all_data_b : TYPE
        DESCRIPTION.
    bin_width : TYPE, optional
        DESCRIPTION. The default is None.
    bin_num : TYPE, optional
        DESCRIPTION. The default is 10.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    override_name_check : TYPE, optional
        DESCRIPTION. The default is False.
    block_length : TYPE, optional
        DESCRIPTION. The default is None.
    cumdist : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    data_names : TYPE
        DESCRIPTION.
    data_jsdist : TYPE
        DESCRIPTION.
    data_kld_ab : TYPE
        DESCRIPTION.
    data_kld_ba : TYPE
        DESCRIPTION.

    """
    """
    Calculates the Jensen-Shannon distance and the Kullback-Leibler divergences for each feature from two ensembles.
    
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
        bin_width : float, default=None
            Bin width for the axis to compare the distributions on. 
            If bin_width is None, bin_num (see below) bins are used and the width is determined from the common histogram.
        bin_num : int, default=10
            Number of bins for the axis to compare the distributions on (only if bin_width=None).
        verbose : bool, default=True
            Print intermediate results.
        override_name_check : bool, default=False
            Only check number of features, not their names.
        block_length : int, optional
            The length of frames to include in each block. The default is None.
        cumdist : bool, optional
            If a cumulative block analysis is desired, set to True. The default is False.   
        
    Returns
    -------
        data_names : list of str
            Feature names.
        data_jsdist : float array
            Jensen-Shannon distance for each feature.
        data_kld_ab : float array
            Kullback-Leibler divergences of data_a wrt to data_b.
        data_kld_ba : float array
            Kullback-Leibler divergences of data_b wrt to data_a.
        
    """
    all_data_a, all_data_b = all_data_a.T, all_data_b.T
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract the names of the features
    data_names = features_a
    # Initialize relative entropy and average value
    data_jsdist = np.zeros(len(data_names))
    data_kld_ab = np.zeros(len(data_names))
    data_kld_ba = np.zeros(len(data_names))
    data_avg    = np.zeros(len(data_names))
    # Loop over all features
    for i in range(len(all_data_a)):
        if block_length is not None:
            if cumdist is True:
                BL1 = block_length[0]
                BL2 = block_length[1]
                data_a = all_data_a[i][BL1:BL2]
                data_b = all_data_b[i][BL1:BL2]
            else:
                data_a = all_data_a[i][:block_length]
                data_b = all_data_b[i][:block_length]                
        else:
            data_a = all_data_a[i]
            data_b = all_data_b[i]
        # Combine both data sets
        data_both = np.concatenate((data_a,data_b))
        data_avg[i] = np.mean(data_both)
        # Get bin values for all histograms from the combined data set
        if bin_width is None:
            bins = bin_num
        else:
            bins_min = np.min( data_both )
            bins_max = np.max( data_both )
            bins = np.arange(bins_min,bins_max,bin_width)
        # Calculate histograms for combined and single data sets
        histo_both = np.histogram(data_both, bins = bins, density = True)
        histo_a = np.histogram(data_a, density = True, bins = histo_both[1])
        distr_a = histo_a[0] / np.sum(histo_a[0])
        histo_b = np.histogram(data_b, density = True, bins = histo_both[1])
        distr_b = histo_b[0] / np.sum(histo_b[0])
        # Calculate relative entropies between the two data sets (Kullback-Leibler divergence)
        data_kld_ab[i] = np.sum( sp.special.kl_div(distr_a,distr_b) )
        data_kld_ba[i] = np.sum( sp.special.kl_div(distr_b,distr_a) )
        # Calculate the Jensen-Shannon distance
        data_jsdist[i] = scipy.spatial.distance.jensenshannon(distr_a, distr_b, base=2.0)
        # Print information
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f %1.2f"%(data_jsdist[i],data_kld_ab[i],data_kld_ba[i]))   
    return data_names, data_jsdist, data_kld_ab, data_kld_ba



def unc_ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbc=True, 
                          verbose=True, write_plots=None, override_name_check=False, block_length=None, bin_no=180, cumdist=False):

    """
    Calculates State Specific Information statistic for a feature across two ensembles.
    
    Parameters
    ----------
    features_a : list of str
        Feature names of the first ensemble. 
    features_b : list of str
        Feature names of the first ensemble. 
        Must be the same as features_a. Provided as a sanity check. 
    all_data_a : float array
        Trajectory data from the first ensemble. Format: [frames,frame_data].
    all_data_b : float array
        Trajectory data from the second ensemble. Format: [frames,frame_data].
    torsions : str
        Torsion angles to use for SSI, including backbone - 'bb', and sidechain - 'sc'. 
        Default is None.
    pocket_occupancy : bool, optional
        Set to 'True' if the data input is pocket occupancy distribution.
        The default is None.
    pbc : bool, optional
        If true, the apply periodic bounary corrections on angular distribution inputs.
        The input for periodic correction must be radians. The default is True.
    verbose : bool, default=True
        Print intermediate results.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    override_name_check : bool, default=False
        Only check number of features, not their names.   
    block_length : int, optional
        The length of frames to include in each block. The default is None.
    bin_no : int, optional
        The number of bins to use in state clustering. The default is 180.
    cumdist : bool, optional
        If a cumulative block analysis is desired, set to True. The default is False.       
        
            
    Returns
    -------
        data_names : list of str
            Feature names.
        data_ssi : float array
            State Specific Information statistics for each feature.

    """
    
    # Get the multivariate timeseries data
    if torsions is None:
         mv_res_feat_a, mv_res_data_a = features_a,all_data_a
         mv_res_feat_b, mv_res_data_b = features_b,all_data_b
    else:
         mv_res_feat_a, mv_res_data_a = get_multivar_res_timeseries(features_a,all_data_a,torsions+'-torsions',write=False,out_name='')
         mv_res_feat_b, mv_res_data_b = get_multivar_res_timeseries(features_b,all_data_b,torsions+'-torsions',write=False,out_name='')
    
         mv_res_feat_a, mv_res_data_a = mv_res_feat_a[torsions+'-torsions'], mv_res_data_a[torsions+'-torsions']
         mv_res_feat_b, mv_res_data_b = mv_res_feat_b[torsions+'-torsions'], mv_res_data_b[torsions+'-torsions']
    
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(mv_res_feat_a) == len(mv_res_feat_b)
    else:
        assert mv_res_feat_a == mv_res_feat_b
    assert mv_res_data_a.shape[0] == mv_res_data_b.shape[0] 
    # Extract the names of the features
    data_names = mv_res_feat_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros(len(data_names))
    # Loop over all features    
    for residue in range(len(mv_res_data_a)):
        data_a = mv_res_data_a[residue]
        data_b = mv_res_data_b[residue]

        combined_dist=[]
        
        if block_length is not None:
            if cumdist is True:
                BL1 = block_length[0]
                BL2 = block_length[1]
                for dist_no in range(len(data_a)):
                    # # # combine the ensembles into one distribution (condition_a + condition_b)
                    data_both = list(data_a[dist_no][BL1:BL2]) + list(data_b[dist_no][BL1:BL2])      
                    combined_dist.append(data_both)
            else:               
                for dist_no in range(len(data_a)):
                    # # # combine the ensembles into one distribution (condition_a + condition_b)
                    data_both = list(data_a[dist_no][:block_length]) + list(data_b[dist_no][:block_length])      
                    combined_dist.append(data_both)
        else:
            for dist_no in range(len(data_a)):
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                data_both = list(data_a[dist_no]) + list(data_b[dist_no])      
                combined_dist.append(data_both)
            traj1_len = len(data_a[dist_no]) 
            

                
            ## Saving distribution length
            traj1_len = len(data_a[dist_no][BL1:BL2])   

        if pbc is True:
            feat_distr = [correct_angle_periodicity(distr) for distr in combined_dist]
        else:
            feat_distr = combined_dist        
    
        if pocket_occupancy is True: 
            ## Define states for water occupancy 
            feat_states = [[-0.5,0.5,1.5]]    
        else:     
            cluster=1
            feat_states=[]
            for dim_num in range(len(feat_distr)):
                if write_plots is True:
                    plot_name = data_names[residue]
                else:
                    plot_name = None
                try:
                    feat_states.append(determine_state_limits(feat_distr[dim_num], 
                                                              traj1_len, 
                                                              write_plots=write_plots, 
                                                              write_name=plot_name,
                                                              gauss_bins=bin_no))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
                    cluster=0
                    
        if cluster==0:
            SSI = -1
            data_ssi[residue] = SSI
            if verbose is True:
                print(data_names[residue],data_ssi[residue])
        else:
            H_feat=calculate_entropy(feat_states,feat_distr) 
                    
            if H_feat != 0:
                ##calculating the entropy for set_distr_b
                ## if no dist (None) then apply the binary dist for two simulations
                ens_distr=[[0.5]*traj1_len + [1.5]*int(len(feat_distr[0])-traj1_len)]
                ens_states= [[0,1,2]]  
                    
                traj_1_fraction = traj1_len/len(feat_distr[0])
                traj_2_fraction = 1 - traj_1_fraction
                norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)
                H_ens = norm_factor
                
                featens_joint_states= feat_states + ens_states
                featens_joint_distr= feat_distr + ens_distr
                H_featens=calculate_entropy(featens_joint_states,featens_joint_distr)
        
                SSI = ((H_feat + H_ens) - H_featens)/norm_factor
            
            data_ssi[residue] = SSI
                
            if verbose is True:
                print(data_names[residue],data_ssi[residue])
            
            
    return data_names, data_ssi


