import pandas as pd
import numpy as np


def write_csv_features(feature_names, feature_data, csv_file):
    """
    Write features to a CSV file.

    Parameters
    ----------
    feature_names : list of str
        Names of the features.
    features_data : numpy array
        Data for the features. Format: [frames, frame_data].
    csv_file : str
        File name for the output CSV file.

    """ 
    df = pd.DataFrame(feature_data, columns=feature_names)
    df.to_csv(csv_file, index=False)
    
    
def read_csv_features(csv_file):
    """
    Load features from a CSV file as produced by PENSA.

    Parameters
    ----------
    csv_file : str
        File name for the input CSV file.

    Returns
    -------
    feature_names : list of str
        Names of the features.
    features_data : numpy array
        Data for the features. Format: [frames, frame_data].

    """ 
    df = pd.read_csv(csv_file)
    feature_names = list(df.keys())
    feature_data = np.zeros([len(df),len(feature_names)])
    for i,f in enumerate(feature_names):
        feature_data[:,i] = df[f]
    return feature_names, feature_data 
    
    
def get_drormd_features(csv_file):
    """
    Load features from a CSV file as produced by DrorMD.

    Parameters
    ----------
    csv_file : str
        File name for the input CSV file.

    Returns
    -------
    feature_names : list of str
        Names of the features.
    features_data : numpy array
        Data for the features. Format: [frames, frame_data].

    """ 
    df = pd.read_csv(csv_file, index_col=0)
    feature_names = list(df.keys())
    feature_data = np.zeros([len(df),len(feature_names)])
    for i,f in enumerate(feature_names):
        feature_data[:,i] = df[f]
    return feature_names, feature_data

