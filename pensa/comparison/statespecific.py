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
        dist_a_len = len(data_a[dist_no])   
         
        if wat_occupancy is True: 
            ## Define states for water occupancy 
            wat_occupancy = [[-0.5,0.5,1.5]]
            
        if write_plots is True:
            write_name = data_names[residue]
            data_ssi[residue] = calculate_ssi(combined_dist,
                                              traj1_len = dist_a_len,
                                              a_states = wat_occupancy,
                                              write_plots=write_plots,
                                              write_name=write_name,
                                              pbc=pbc)
        else:
            data_ssi[residue] = calculate_ssi(combined_dist,
                                              traj1_len = dist_a_len,
                                              a_states = wat_occupancy,
                                              pbc=pbc)
            
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

        res1_data_ens1 = mv_res_data_a[res1]
        res1_data_ens2 = mv_res_data_b[res1]
        res1_combined_dist=[]
        for dist_no in range(len(res1_data_ens1)):
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = list(res1_data_ens1[dist_no]) + list(res1_data_ens2[dist_no])     
            res1_combined_dist.append(res1_data_both)
                
        for res2 in range(res1, len(mv_res_data_a)):
                
            res2_data_ens1 = mv_res_data_a[res2]
            res2_data_ens2 = mv_res_data_b[res2]     
            res2_combined_dist=[]
            for dist_no in range(len(res2_data_ens1)):
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                res2_data_both = list(res2_data_ens1[dist_no]) + list(res2_data_ens2[dist_no])
                res2_combined_dist.append(res2_data_both)            
            
            ## Saving distribution length
            traj1_len = len(res2_data_ens1[dist_no])               
            
            data_ssi[res1][res2] = calculate_ssi(distr_a_input = res1_combined_dist,
                                                 traj1_len = traj1_len,
                                                 distr_b_input = res2_combined_dist)
            
            data_ssi[res2][res1] = data_ssi[res1][res2]    
                
                
            if verbose is True:
                print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])
    
    return data_names, data_ssi


def cossi_featens_analysis(features_a, features_b, 
                           all_data_a, all_data_b, 
                           cossi_features_a, cossi_features_b, 
                           cossi_all_data_a, cossi_all_data_b, 
                           torsions='bb', verbose=True, override_name_check=False):

    """
    Calculates State Specific Information CoSSI statistic between
    two features and the binary ensemble change.
    

    Parameters
    ----------
    
    features_a : list of str
        Feature names of the first ensemble. 
    features_b : list of str
        Feature names of the first ensemble. 
        Must be the same as features_a. Provided as a sanity check. 
    all_data_a : float array
        Trajectory data from the first ensemble.
    all_data_b : float array
        Trajectory data from the second ensemble. 
    cossi_features_a : list of str
        Feature names of the first ensemble. 
    cossi_features_b : list of str
        Feature names of the first ensemble. 
        Must be the same as features_a. Provided as a sanity check. 
    cossi_all_data_a : float array
        Trajectory data from the first ensemble. 
    cossi_all_data_b : float array
        Trajectory data from the second ensemble. 
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
    cossi_data_names : list of str
        Feature names of stabilizing feature.
    data_cossi : float array
        State Specific Information Co-SSI statistics for each feature.

    """
    
    # Get the multivariate timeseries data
    mv1_res_feat_a, mv1_res_data_a = get_multivar_res_timeseries(features_a,all_data_a,torsions+'-torsions',write=False,out_name='')
    mv1_res_feat_b, mv1_res_data_b = get_multivar_res_timeseries(features_b,all_data_b,torsions+'-torsions',write=False,out_name='')
    
    mv2_res_feat_a, mv2_res_data_a = get_multivar_res_timeseries(cossi_features_a,cossi_all_data_a,torsions+'-torsions',write=False,out_name='')
    mv2_res_feat_b, mv2_res_data_b = get_multivar_res_timeseries(cossi_features_b,cossi_all_data_b,torsions+'-torsions',write=False,out_name='')

    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(mv1_res_feat_a) == len(mv1_res_feat_b)
    else:
        assert mv1_res_feat_a == mv1_res_feat_b
    assert mv1_res_data_a.shape[0] == mv1_res_data_b.shape[0] 
    # Extract the names of the features
    data_names = mv1_res_feat_a

    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(mv2_res_feat_a) == len(mv2_res_feat_b)
    else:
        assert mv2_res_feat_a == mv2_res_feat_b
    assert mv2_res_data_a.shape[0] == mv2_res_data_b.shape[0] 
    # Extract the names of the features
    cossi_data_names = mv2_res_feat_a
    
    # Initialize relative entropy and average value
    data_ssi = np.zeros((len(data_names),len(cossi_data_names)))
    data_cossi = np.zeros((len(data_names),len(cossi_data_names)))
    for res1 in range(len(mv1_res_data_a)):

        res1_data_ens1 = mv1_res_data_a[res1]
        res1_data_ens2 = mv1_res_data_b[res1]
        res1_combined_dist=[]
        for dist_no in range(len(res1_data_ens1)):
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = list(res1_data_ens1[dist_no]) + list(res1_data_ens2[dist_no])  
            res1_combined_dist.append(res1_data_both)
                
        for res2 in range(len(mv2_res_data_a)):
                
            res2_data_ens1 = mv2_res_data_a[res2]
            res2_data_ens2 = mv2_res_data_b[res2]     
            res2_combined_dist=[]
            for dist_no in range(len(res2_data_ens1)):
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                res2_data_both = list(res2_data_ens1[dist_no]) + list(res2_data_ens2[dist_no])   
                res2_combined_dist.append(res2_data_both)
                
            ## Saving distribution length
            traj1_len = len(res2_data_ens1[dist_no])        
            
            data_ssi[res1][res2], data_cossi[res1][res2] = calculate_cossi(res1_combined_dist,
                                                                           traj1_len,
                                                                           res2_combined_dist)
                
            if verbose is True:
                print('\nFeature Pair: ', data_names[res1], cossi_data_names[res2],
                      '\nSSI[bits]: ', data_ssi[res1][res2],
                      '\nCo-SSI[bits]: ', data_cossi[res1][res2])
    
    return data_names, cossi_data_names, data_ssi, data_cossi


