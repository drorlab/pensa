import numpy as np
from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *



def ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, wat_occupancy=None, pbc=True, 
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
    
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract the names of the features
    data_names = features_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros(len(data_names))
    # Loop over all features    
    for residue in tqdm(range(len(all_data_a))):
        data_a = all_data_a[residue]
        data_b = all_data_b[residue]

        combined_dist=[]
        for dist_no in range(len(data_a)):
            # # # Make sure the ensembles have the same length of trajectory
            sim1,sim2=match_sim_lengths(list(data_a[dist_no]),list(data_b[dist_no]))
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            data_both = sim1+sim2      
            combined_dist.append(data_both)
            
        if wat_occupancy is True: 
            ## Define states for water occupancy 
            wat_occupancy = [[-0.5,0.5,1.5]]
            
        if write_plots is True:
            write_name = data_names[residue]
            data_ssi[residue] = calculate_ssi(combined_dist,
                                              a_states = wat_occupancy,
                                              write_plots=write_plots,
                                              write_name=write_name,
                                              pbc=pbc)
        else:
            data_ssi[residue] = calculate_ssi(combined_dist,
                                              a_states = wat_occupancy,
                                              pbc=pbc)
            
        if verbose is True:
            print(data_names[residue],data_ssi[residue])
        
    return data_names, data_ssi


def ssi_feature_analysis(features_a, features_b, all_data_a, all_data_b, verbose=True, override_name_check=False):

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
    
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract the names of the features
    data_names = features_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros((len(data_names),len(data_names)))
    # Loop over all features
    
    for res1 in range(len(all_data_a)):

        res1_data_ens1 = all_data_a[res1]
        res1_data_ens2 = all_data_b[res1]
        res1_combined_dist=[]
        for dist_no in range(len(res1_data_ens1)):
            # # # Make sure the ensembles have the same length of trajectory
            sim1,sim2=match_sim_lengths(list(res1_data_ens1[dist_no]),list(res1_data_ens2[dist_no]))
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = sim1+sim2      
            res1_combined_dist.append(res1_data_both)
                
        for res2 in range(res1, len(all_data_a)):
                
            res2_data_ens1 = all_data_a[res2]
            res2_data_ens2 = all_data_b[res2]     
            res2_combined_dist=[]
            for dist_no in range(len(res2_data_ens1)):
                # # # Make sure the ensembles have the same length of trajectory
                sim1,sim2=match_sim_lengths(list(res2_data_ens1[dist_no]),list(res2_data_ens2[dist_no]))
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                res2_data_both = sim1+sim2      
                res2_combined_dist.append(res2_data_both)            
            
            data_ssi[res1][res2] = calculate_ssi(res1_combined_dist,
                                                 res2_combined_dist)
            
            data_ssi[res2][res1] = data_ssi[res1][res2]    
                
                
            if verbose is True:
                print('SSI[bits]: ',data_names[res1],data_names[res2],data_ssi[res1][res2])
    
    return data_names, data_ssi


def cossi_featens_analysis(features_a, features_b, 
                           all_data_a, all_data_b, 
                           cossi_features_a, cossi_features_b, 
                           cossi_all_data_a, cossi_all_data_b, 
                           verbose=True, override_name_check=False):

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

    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract the names of the features
    data_names = features_a

    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(cossi_features_a) == len(cossi_features_b)
    else:
        assert cossi_features_a == cossi_features_b
    assert cossi_all_data_a.shape[0] == cossi_all_data_b.shape[0] 
    # Extract the names of the features
    cossi_data_names = cossi_features_a
    
    # Initialize relative entropy and average value
    data_ssi = np.zeros((len(data_names),len(cossi_data_names)))
    data_cossi = np.zeros((len(data_names),len(cossi_data_names)))
    for res1 in range(len(all_data_a)):

        res1_data_ens1 = all_data_a[res1]
        res1_data_ens2 = all_data_b[res1]
        res1_combined_dist=[]
        for dist_no in range(len(res1_data_ens1)):
            # # # Make sure the ensembles have the same length of trajectory
            sim1,sim2=match_sim_lengths(list(res1_data_ens1[dist_no]),list(res1_data_ens2[dist_no]))
            # # # combine the ensembles into one distribution (condition_a + condition_b)
            res1_data_both = sim1+sim2      
            res1_combined_dist.append(res1_data_both)
                
        for res2 in range(len(cossi_all_data_a)):
                
            res2_data_ens1 = cossi_all_data_a[res2]
            res2_data_ens2 = cossi_all_data_b[res2]     
            res2_combined_dist=[]
            for dist_no in range(len(res2_data_ens1)):
                # # # Make sure the ensembles have the same length of trajectory
                sim1,sim2=match_sim_lengths(list(res2_data_ens1[dist_no]),list(res2_data_ens2[dist_no]))
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                res2_data_both = sim1+sim2      
                res2_combined_dist.append(res2_data_both)
            
            data_ssi[res1][res2], data_cossi[res1][res2] = calculate_cossi(res1_combined_dist,
                                                                           res2_combined_dist)
                
            if verbose is True:
                print('\nFeature Pair: ', data_names[res1], cossi_data_names[res2],
                      '\nSSI[bits]: ', data_ssi[res1][res2],
                      '\nCo-SSI[bits]: ', data_cossi[res1][res2])
    
    return data_names, cossi_data_names, data_ssi, data_cossi


