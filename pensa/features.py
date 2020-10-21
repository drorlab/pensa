#import mdshare
import pyemma
from pyemma.util.contexts import settings


# -- Loading the Features --


def get_features(pdb, xtc, start_frame=0):
    """
    Load the features. Currently implemented: bb-torsions, sc-torsions, bb-distances
    http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html
    
    Args:
        pdb (str): File name for the reference file (PDB or GRO format).
        xtc (str): File name for the trajectory (xtc format).
        start_frame (int, optional): First frame to read the features. Defaults to 0.
    
    Returns:
        feature_names (dict): Names of all features
        features_data (dict): Data for all features
    
    """
    # Initialize the dictionaries.
    feature_names = {}
    features_data = {}
    # Add backbone torsions.
    bbtorsions_feat = pyemma.coordinates.featurizer(pdb)
    bbtorsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    bbtorsions_data = pyemma.coordinates.load(xtc, features=bbtorsions_feat)[start_frame:]
    feature_names['bb-torsions'] = bbtorsions_feat
    features_data['bb-torsions'] = bbtorsions_data
    # Add sidechain torsions.
    sctorsions_feat = pyemma.coordinates.featurizer(pdb)
    sctorsions_feat.add_sidechain_torsions(cossin=True, periodic=False)
    sctorsions_data = pyemma.coordinates.load(xtc, features=sctorsions_feat)[start_frame:]
    feature_names['sc-torsions'] = sctorsions_feat
    features_data['sc-torsions'] = sctorsions_data
    # Add backbone C-alpha distances.
    bbdistances_feat = pyemma.coordinates.featurizer(pdb)
    bbdistances_feat.add_distances(bbdistances_feat.pairs(bbdistances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
    bbdistances_data = pyemma.coordinates.load(xtc, features=bbdistances_feat)[start_frame:]
    feature_names['bb-distances'] = bbdistances_feat
    features_data['bb-distances'] = bbdistances_data
    # Return the dictionaries.
    return feature_names, features_data

