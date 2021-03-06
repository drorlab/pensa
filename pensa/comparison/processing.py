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




def get_feature_timeseries(feat, data, feature_type, feature_name):
    """
    Returns the timeseries of one particular feature.
    
    Parameters
    ----------
        feat : list of str
            List with all feature names.
        data : float array
            Feature values data from the simulation.
        feature_type : str
            Type of the selected feature 
            ('bb-torsions', 'bb-distances', 'sc-torsions').
        feature_name : str
           Name of the selected feature.
    
    Returns
    -------
        timeseries : float array
            Value of the feature for each frame.
    
    """
    # Select the feature and get its index.
    index = np.where( np.array( feat[feature_type] ) == feature_name )[0][0]
    # Extract the timeseries.
    timeseries = data[feature_type][:,index]
    return timeseries


def multivar_res_timeseries_data(feat, data, feature_type, write=None, out_name=None):
    """
    

    Parameters
    ----------
    feat : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    feature_type : TYPE
        DESCRIPTION.
    write : TYPE, optional
        DESCRIPTION. The default is None.
    out_name : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    feature_names : TYPE
        DESCRIPTION.
    features_data : TYPE
        DESCRIPTION.

    """
    
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
    
    feat_name_list = feat[feature_type]
    #obtaining the residue numbers 
    res_numbers = [int(feat_name.split()[-1]) for feat_name in feat_name_list]
    #grouping indices where feature refers to same residue
    index_same_res = [list(np.where(np.array(res_numbers)==seq_num)[0])
                      for seq_num in list(set(res_numbers))]   
    #obtaining timeseries data for each residue
    multivar_res_timeseries_data=[]
    sorted_names = []
    for residue in range(len(index_same_res)):
        feat_timeseries=[]
        for residue_dim in index_same_res[residue]:
            single_feat_timeseries = get_feature_timeseries(feat,data,feature_type,feat_name_list[residue_dim])            
            feat_timeseries.append(list(single_feat_timeseries))
        multivar_res_timeseries_data.append(feat_timeseries)
        resname= ''.join(feat_name_list[residue_dim].split()[-2:])
        sorted_names.append(resname)
        if write is True:
            for subdir in [feature_type+'/']:
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
            filename= feature_type+'/' + out_name + resname + ".txt"
            np.savetxt(filename, feat_timeseries, delimiter=',', newline='\n')
            
    # return multivar_res_timeseries_data
    feature_names[feature_type]=sorted_names
    features_data[feature_type]=np.array(multivar_res_timeseries_data, dtype=object)
    # Return the dictionaries.
    return feature_names, features_data            


def sort_features(names, sortby):
    """
    Sorts features by a list of values.
    
    Parameters
    ----------
        names : str array
            Array of feature names.
        sortby : float array
            Array of the values to sort the names by.
        
    Returns
    -------
        sort : array of tuples [str,float]
            Array of sorted tuples with feature and value.
    
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



