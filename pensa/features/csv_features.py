import pandas as pd
import numpy as np


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
        Names of all C-alpha distances.
    features_data : numpy array
        Data for all C-alpha distances [Å].

    """ 
    df = pd.read_csv(csv_file, index_col=0)
    feature_names = list(df.keys())
    feature_data = np.zeros([len(df),len(feature_names)])
    for i,f in enumerate(feature_names):
        feature_data[:,i] = df[f]
    return feature_names, feature_data

