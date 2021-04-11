import numpy as np
from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *



def ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, wat_occupancy=None, pbc=True, 
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
    wat_occupancy : bool, optional
        Set to 'True' if the data input is water pocket occupancy distribution.
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
         
        if wat_occupancy is True: 
            ## Define states for water occupancy 
            wat_occupancy = [[-0.5,0.5,1.5]]            

        if pbc is True:
            feat_distr = [correct_angle_periodicity(distr) for distr in combined_dist]
        else:
            feat_distr = combined_dist        
    
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
    data_names = mv_res_feat_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros((len(data_names),len(data_names)))
    # Loop over all features
    
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
                                    
                    data_ssi[res1][res2], data_ssi[res2][res1] = SSI, SSI       
                            
                    if verbose is True:
                        print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])        
                else:
                    data_ssi[res1][res2], data_ssi[res2][res1] = 0, 0
                    if verbose is True:
                        print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])    
                    
        else:
            for res2 in range(res1+1, len(mv_res_data_a)):
                data_ssi[res1][res2], data_ssi[res2][res1] = 0, 0
                if verbose is True:
                    print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])    
                
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
    data_names = mv_res_feat_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros((len(data_names),len(data_names)))
    data_cossi = np.zeros((len(data_names),len(data_names)))
    # Loop over all features
    
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
             
                    traj_1_fraction = traj1_len/len(set_distr_a[0])
                    traj_2_fraction = 1 - traj_1_fraction                    
                    norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)
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
                    
                    data_ssi[res1][res2], data_ssi[res2][res1] = SSI, SSI       
                    data_cossi[res1][res2], data_cossi[res2][res1] = coSSI, coSSI       
                    
                    
                            
                    if verbose is True:
                        print('\nFeature Pair: ', data_names[res1], data_names[res2],
                              '\nSSI[bits]: ', data_ssi[res1][res2],
                              '\nCo-SSI[bits]: ', data_cossi[res1][res2])
         
                else:
                    data_ssi[res1][res2], data_ssi[res2][res1] = 0, 0
                    data_cossi[res1][res2], data_cossi[res2][res1] = 0, 0     
                    if verbose is True:
                        print('\nFeature Pair: ', data_names[res1], data_names[res2],
                              '\nSSI[bits]: ', data_ssi[res1][res2],
                              '\nCo-SSI[bits]: ', data_cossi[res1][res2])
    
                    
        else:
            for res2 in range(res1+1, len(mv_res_data_a)):
                data_ssi[res1][res2], data_ssi[res2][res1] = 0, 0
                data_cossi[res1][res2], data_cossi[res2][res1] = 0, 0     
                if verbose is True:
                    print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])    
                if verbose is True:
                    print('\nFeature Pair: ', data_names[res1], data_names[res2],
                          '\nSSI[bits]: ', data_ssi[res1][res2],
                          '\nCo-SSI[bits]: ', data_cossi[res1][res2])
    
    return data_names, data_ssi, data_cossi
                