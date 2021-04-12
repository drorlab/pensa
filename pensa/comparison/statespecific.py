import numpy as np
from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *

# -- Functions to calculate SSI statistics across paired ensembles --


def ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbc=True, 
                          verbose=True, write_plots=None, override_name_check=False):
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
        for dist_no in range(len(data_a)):
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            data_both = list(data_a[dist_no]) + list(data_b[dist_no])      
            combined_dist.append(data_both)
            
        ## Saving distribution length
        traj1_len = len(data_a[dist_no])   

        if pbc is True:
            feat_distr = [correct_angle_periodicity(distr) for distr in combined_dist]
        else:
            feat_distr = combined_dist        
    
        if pocket_occupancy is True: 
            ## Define states for water occupancy 
            feat_states = [[-0.5,0.5,1.5]]    
        else:     
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
                                                              write_name=plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')

            
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
        
        else: 
            
            SSI = 0
        
        data_ssi[residue] = SSI
            
        if verbose is True:
            print(data_names[residue],data_ssi[residue])
        
    return data_names, data_ssi


def ssi_feature_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, verbose=True, override_name_check=False):

    """
    Calculates State Specific Information statistic between two features across two ensembles.
    
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
    verbose : bool, default=True
        Print intermediate results.
    override_name_check : bool, default=False
        Only check number of features, not their names.
        
        
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
    data_names = []
    for feat1 in range(len(mv_res_feat_a)):
        for feat2 in range(feat1, len(mv_res_feat_a)):
            data_names.append(torsions + ' ' + mv_res_feat_a[feat1] + ' & ' + torsions + ' ' + mv_res_feat_a[feat2])
    # Initialize SSI
    data_ssi = np.zeros(len(data_names))
    # Loop over all features
    count=0
    for res1 in range(len(mv_res_data_a)):
        # print(res1)
        res1_data_ens1 = mv_res_data_a[res1]
        res1_data_ens2 = mv_res_data_b[res1]
        res1_combined_dist=[]
        for dist_no_a in range(len(res1_data_ens1)):
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = list(res1_data_ens1[dist_no_a]) + list(res1_data_ens2[dist_no_a])     
            res1_combined_dist.append(res1_data_both)

        ## Saving distribution length
        traj1_len = len(res1_data_ens1[dist_no_a])   
            
        # if calculate_ssi(res1_combined_dist, traj1_len)!=0:      
        set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in res1_combined_dist]
    
        set_a_states=[]
        for dim_num_a in range(len(set_distr_a)):
                set_a_states.append(determine_state_limits(set_distr_a[dim_num_a], traj1_len))
        H_a=calculate_entropy(set_a_states,set_distr_a) 
        if H_a != 0:

        
            for res2 in range(res1, len(mv_res_data_a)):
                # Only run SSI if entropy is non-zero
                res2_data_ens1 = mv_res_data_a[res2]
                res2_data_ens2 = mv_res_data_b[res2]     
                res2_combined_dist=[]
                for dist_no_b in range(len(res2_data_ens1)):
                    # # # combine the ensembles into one distribution (condition_a + condition_b)
                    res2_data_both = list(res2_data_ens1[dist_no_b]) + list(res2_data_ens2[dist_no_b])
                    res2_combined_dist.append(res2_data_both)            
                 
                set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in res2_combined_dist]
                    
                set_b_states=[]
                for dim_num_b in range(len(set_distr_b)):
                        set_b_states.append(determine_state_limits(set_distr_b[dim_num_b], traj1_len))
                H_b=calculate_entropy(set_b_states,set_distr_b)
                
                if H_b!=0:
                
                    ab_joint_states= set_a_states + set_b_states
                    ab_joint_distributions= set_distr_a + set_distr_b
                    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
            
                    traj_1_fraction = traj1_len/len(set_distr_a[0])
                    traj_2_fraction = 1 - traj_1_fraction
                    norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)
            
                    SSI = ((H_a + H_b) - H_ab)/norm_factor
                                    
                    
                    data_ssi[count] = SSI       
                    if verbose is True:
                        print(data_names[count],'\nSSI[bits]: ',data_ssi[count])    
                    count+=1
                else:
                    if verbose is True:
                        print(data_names[count],'\nSSI[bits]: ',data_ssi[count])    
                    count+=1

        else:
            for res2 in range(res1+1, len(mv_res_data_a)):
                if verbose is True:
                    print(data_names[count],'\nSSI[bits]: ',data_ssi[count])    
                count+=1
                
    return data_names, data_ssi



def cossi_featens_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, verbose=True, override_name_check=False):

    """
    Calculates State Specific Information Co-SSI statistic between two features and the ensembles condition.
    
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
    verbose : bool, default=True
        Print intermediate results.
    override_name_check : bool, default=False
        Only check number of features, not their names.
        
        
    Returns
    -------
        data_names : list of str
            Feature names.
        data_ssi : float array
            State Specific Information SSI statistics for each feature.
        data_cossi : float array
            State Specific Information Co-SSI statistics for each feature.

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
    data_names = []
    for feat1 in range(len(mv_res_feat_a)):
        for feat2 in range(feat1, len(mv_res_feat_a)):
            data_names.append(torsions + ' ' + mv_res_feat_a[feat1] + ' & ' + torsions + ' ' + mv_res_feat_a[feat2])
    # Initialize SSI and Co-SSI
    data_ssi = np.zeros(len(data_names))
    data_cossi = np.zeros(len(data_names))
    # Loop over all features
    count=0    
    for res1 in range(len(mv_res_data_a)):
        # print(res1)
        res1_data_ens1 = mv_res_data_a[res1]
        res1_data_ens2 = mv_res_data_b[res1]
        res1_combined_dist=[]
        for dist_no_a in range(len(res1_data_ens1)):
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = list(res1_data_ens1[dist_no_a]) + list(res1_data_ens2[dist_no_a])     
            res1_combined_dist.append(res1_data_both)

        ## Saving distribution length
        traj1_len = len(res1_data_ens1[dist_no_a])   
            
        # if calculate_ssi(res1_combined_dist, traj1_len)!=0:      
        set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in res1_combined_dist]
    
        set_a_states=[]
        for dim_num_a in range(len(set_distr_a)):
                set_a_states.append(determine_state_limits(set_distr_a[dim_num_a], traj1_len))
        H_a=calculate_entropy(set_a_states,set_distr_a) 
        if H_a != 0:

        
            for res2 in range(res1, len(mv_res_data_a)):
                # Only run SSI if entropy is non-zero
                res2_data_ens1 = mv_res_data_a[res2]
                res2_data_ens2 = mv_res_data_b[res2]     
                res2_combined_dist=[]
                for dist_no_b in range(len(res2_data_ens1)):
                    # # # combine the ensembles into one distribution (condition_a + condition_b)
                    res2_data_both = list(res2_data_ens1[dist_no_b]) + list(res2_data_ens2[dist_no_b])
                    res2_combined_dist.append(res2_data_both)            
                 
                set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in res2_combined_dist]
                    
                set_b_states=[]
                for dim_num_b in range(len(set_distr_b)):
                        set_b_states.append(determine_state_limits(set_distr_b[dim_num_b], traj1_len))
                H_b=calculate_entropy(set_b_states,set_distr_b)
                
                if H_b!=0:
                
                    ab_joint_states= set_a_states + set_b_states
                    ab_joint_distributions= set_distr_a + set_distr_b
                    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
            
                    traj_1_fraction = traj1_len/len(set_distr_a[0])
                    traj_2_fraction = 1 - traj_1_fraction
                    norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)
                    
                    set_distr_c=[[0.5]*traj1_len + [1.5]*int(len(set_distr_a[0])-traj1_len)]
                    set_c_states= [[0,1,2]]                      
                    H_c = norm_factor       
                    
                    ##----------------
                    ab_joint_states = set_a_states + set_b_states
                    ab_joint_distributions = set_distr_a + set_distr_b
                    
                    H_ab = calculate_entropy(ab_joint_states, ab_joint_distributions)
                    ##----------------
                    ac_joint_states =  set_a_states + set_c_states 
                    ac_joint_distributions = set_distr_a + set_distr_c
                    
                    H_ac = calculate_entropy(ac_joint_states, ac_joint_distributions)
                    ##----------------
                    bc_joint_states = set_b_states + set_c_states 
                    bc_joint_distributions = set_distr_b + set_distr_c
                    
                    H_bc = calculate_entropy(bc_joint_states, bc_joint_distributions)
                    ##----------------
                    abc_joint_states = set_a_states + set_b_states + set_c_states 
                    abc_joint_distributions = set_distr_a + set_distr_b + set_distr_c
                    
                    H_abc = calculate_entropy(abc_joint_states, abc_joint_distributions)    
            
                    SSI = ((H_a + H_b) - H_ab)/norm_factor
                    coSSI = ((H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc)/norm_factor     
                    
                    data_ssi[count] = SSI       
                    data_cossi[count] = coSSI       
                    if verbose is True:
                        print('\nFeature Pair: ', data_names[count],
                              '\nSSI[bits]: ', data_ssi[count],
                              '\nCo-SSI[bits]: ', data_cossi[count])
                    count+=1                
         
                else:
                    if verbose is True:
                        print('\nFeature Pair: ', data_names[count],
                              '\nSSI[bits]: ', data_ssi[count],
                              '\nCo-SSI[bits]: ', data_cossi[count])
                    count+=1
                    
        else:
            for res2 in range(res1+1, len(mv_res_data_a)):
                if verbose is True:
                    print('\nFeature Pair: ', data_names[count],
                          '\nSSI[bits]: ', data_ssi[count],
                          '\nCo-SSI[bits]: ', data_cossi[count])
                count+=1

    
    return data_names, data_ssi, data_cossi
                


# -- Functions with more customizable capabilities for users to adapt to their needs --



def _calculate_ssi(distr_a_input, traj1_len, distr_b_input=None, a_states=None, b_states=None,
                  gauss_bins=180, gauss_smooth=None, pbc=True, write_plots=None, write_name=None):
    """
    Calculates the State Specific Information SSI [bits] between two features from two ensembles. 
    By default, the second feature is the binary switch between ensembles.

    SSI(a,b) = H_a + H_b - H_ab
    H = Conformational state entropy

    Parameters
    ----------
    distr_a_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water
    distr_b_input : list of lists, optional
        A list containing multivariate distributions (lists) for a particular
        residue or water. The default is None and a binary switch is assigned.
    a_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    b_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    gauss_bins : int, optional
        Number of histogram bins to assign for the clustering algorithm. 
        The default is 180.
    gauss_smooth : int, optional
        Number of bins to perform smoothing over. The default is ~10% of gauss_bins.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.

    Returns
    -------
    SSI : float
        State Specific Information (SSI[bits]) shared between input a and input b (default is binary switch).

    """
    
    
    try:
        ##calculating the entropy for set_distr_a
        ## if set_distr_a only contains one distributions
        if pbc is True:
            if type(distr_a_input[0]) is not list:
                set_distr_a=[correct_angle_periodicity(distr_a_input)]
            ## else set_distr_a is a nested list of multiple distributions (bivariate)
            else:
                set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in distr_a_input]
        else:
            set_distr_a=distr_a_input        
        
        if a_states is None:    
            set_a_states=[]
            for dim_num in range(len(set_distr_a)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_a_states.append(determine_state_limits(set_distr_a[dim_num], traj1_len, gauss_bins, gauss_smooth, write_plots, plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_a_states = a_states
            
        H_a=calculate_entropy(set_a_states,set_distr_a) 
                
        ##calculating the entropy for set_distr_b
        ## if no dist (None) then apply the binary dist for two simulations
        if distr_b_input is None:       
            set_distr_b=[[0.5]*traj1_len + [1.5]*int(len(set_distr_a[0])-traj1_len)]
            set_b_states= [[0,1,2]]  
            
        else:
            if pbc is True:
                if type(distr_b_input[0]) is not list:
                    set_distr_b=[correct_angle_periodicity(distr_b_input)]
                else:
                    set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in distr_b_input]
            else:
                set_distr_b=distr_b_input
                
            if b_states is None:    
                set_b_states=[]
                for dim_num in range(len(set_distr_b)):
                    if write_name is not None:
                        plot_name = write_name + '_dist' + str(dim_num)
                    else:
                        plot_name = None
                    try:
                        set_b_states.append(determine_state_limits(set_distr_b[dim_num], traj1_len, gauss_bins, gauss_smooth, write_plots, plot_name))
                    except:    
                        print('Distribution B not clustering properly.\nTry altering Gaussian parameters or input custom states.')
                
            else:
                set_b_states = b_states 
        H_b=calculate_entropy(set_b_states,set_distr_b)
    
        ab_joint_states= set_a_states + set_b_states
        ab_joint_distributions= set_distr_a + set_distr_b
        H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)

        traj_1_fraction = traj1_len/len(set_distr_a[0])
        traj_2_fraction = 1 - traj_1_fraction
        norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)

        SSI = ((H_a + H_b) - H_ab)/norm_factor
            
    except:
        SSI = -1
        if write_name is not None:
            print('WARNING: Input error for ' + write_name)
        else:
            print('WARNING: Input error')
            
        print('Default output of SSI= -1.')
        
    return round(SSI,4)


def _calculate_cossi(distr_a_input, traj1_len, distr_b_input, distr_c_input=None, a_states=None, b_states=None,
                    c_states=None, gauss_bins=180, gauss_smooth=None, write_plots=None,write_name=None):
    """
    Calculates the State Specific Information Co-SSI [bits] between three features from two ensembles. 
    By default, the third feature is the binary switch between ensembles.
    
    CoSSI(a,b,c) = H_a + H_b + H_c - H_ab - H_bc - H_ac + H_abc
    
    H = Conformational state entropy
    
    Parameters
    ----------
        

    distr_a_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water
    distr_b_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water. 
    distr_c_input : list of lists, optional
        A list containing multivariate distributions (lists) for a particular
        residue or water. The default is None and a binary switch is assigned.
    a_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    b_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    c_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    gauss_bins : int, optional
        Number of histogram bins to assign for the clustering algorithm. 
        The default is 180.
    gauss_smooth : int, optional
        Number of bins to perform smoothing over. The default is ~10% of gauss_bins.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.

    Returns
    -------
    SSI : float
        SSI[bits] shared between input a and input b (default is binary switch).
    coSSI : float
        Co-SSI[bits] shared between input a, input b and input c (default is binary switch).

    """     
 
    try:       
        ##calculating the entropy for set_distr_a
        ## if set_distr_a only contains one distributions
        if type(distr_a_input[0]) is not list:
            set_distr_a=[correct_angle_periodicity(distr_a_input)]
        ## else set_distr_a is a nested list of multiple distributions (bivariate)
        else:
            set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in distr_a_input]
        
        if a_states is None:    
            set_a_states=[]
            for dim_num in range(len(set_distr_a)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_a_states.append(determine_state_limits(set_distr_a[dim_num],
                                                               traj1_len,
                                                               gauss_bins, 
                                                               gauss_smooth, 
                                                               write_plots, 
                                                               plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_a_states = a_states
            
        H_a=calculate_entropy(set_a_states,set_distr_a)     

        ##----------------
        ##calculating the entropy for set_distr_b
        if type(distr_b_input[0]) is not list:
            set_distr_b=[correct_angle_periodicity(distr_b_input)]
        ## else set_distr_b is a nested list of multiple distributions (bivariate)
        else:
            set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in distr_b_input]
        
        if b_states is None:    
            set_b_states=[]
            for dim_num in range(len(set_distr_b)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_b_states.append(determine_state_limits(set_distr_b[dim_num],
                                                               traj1_len,
                                                               gauss_bins, 
                                                               gauss_smooth, 
                                                               write_plots, 
                                                               plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_b_states = b_states
            
        H_b=calculate_entropy(set_b_states,set_distr_b)     
        
        ##----------------
        ##calculating the entropy for set_distr_c
        ## if no dist (None) then apply the binary dist for two simulations
        if distr_c_input is None:
            set_distr_c=[[0.5]*traj1_len + [1.5]*int(len(set_distr_a[0])-traj1_len)]
            set_c_states= [[0,1,2]]  
            
            
        else:
            if type(distr_c_input[0]) is not list:
                set_distr_c=[correct_angle_periodicity(distr_c_input)]
            else:
                set_distr_c=[correct_angle_periodicity(distr_c) for distr_c in distr_c_input]
            if c_states is None:    
                set_c_states=[]
                for dim_num in range(len(set_distr_c)):
                    if write_name is not None:
                        plot_name = write_name + '_dist' + str(dim_num)
                    else:
                        plot_name = None
                    try:
                        set_c_states.append(determine_state_limits(set_distr_c[dim_num], traj1_len, gauss_bins, gauss_smooth, write_plots, plot_name))
                    except:    
                        print('Distribution C not clustering properly.\nTry altering Gaussian parameters or input custom states.')
            else:
                set_c_states = c_states
        H_c=calculate_entropy(set_c_states,set_distr_c)
    
        ##----------------
        ab_joint_states = set_a_states + set_b_states
        ab_joint_distributions = set_distr_a + set_distr_b
        
        H_ab = calculate_entropy(ab_joint_states, ab_joint_distributions)
        ##----------------
        ac_joint_states =  set_a_states + set_c_states 
        ac_joint_distributions = set_distr_a + set_distr_c
        
        H_ac = calculate_entropy(ac_joint_states, ac_joint_distributions)
        ##----------------
        bc_joint_states = set_b_states + set_c_states 
        bc_joint_distributions = set_distr_b + set_distr_c
        
        H_bc = calculate_entropy(bc_joint_states, bc_joint_distributions)
        ##----------------
        abc_joint_states = set_a_states + set_b_states + set_c_states 
        abc_joint_distributions = set_distr_a + set_distr_b + set_distr_c
        
        H_abc = calculate_entropy(abc_joint_states, abc_joint_distributions)    
 
        traj_1_fraction = traj1_len/len(set_distr_a[0])
        traj_2_fraction = 1 - traj_1_fraction
        norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)

        SSI = ((H_a + H_b) - H_ab)/norm_factor
        coSSI = ((H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc)/norm_factor
              
        ##conditional mutual info for sanity check
        # con_mut_inf = H_ac + H_bc - H_c - H_abc
        
    except:
        SSI = -1
        coSSI = -1
        
        if write_name is not None:
            print('WARNING: Error for ' + write_name)
        else:
            print('WARNING: Error')
            
        print('Default output of -1.')   
        
    return round(SSI,4), round(coSSI,4)



