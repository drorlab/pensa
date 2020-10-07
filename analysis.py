import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt

# -- Import all analysis methods --

from pensa.pca import *
from pensa.clusters import *
from pensa.featurediff import *


# -- Loading the Features --

def get_features(pdb,xtc,start_frame):
    """
    Load the features. Currently implemented: bb-torsions, sc-torsions, bb-distances
    http://www.emma-project.org/latest/api/generated/pyemma.coordinates.featurizer.html
    """
    
    feature_names = {}
    features_data = {}

    bbtorsions_feat = pyemma.coordinates.featurizer(pdb)
    bbtorsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    bbtorsions_data = pyemma.coordinates.load(xtc, features=bbtorsions_feat)[start_frame:]
    feature_names['bb-torsions'] = bbtorsions_feat
    features_data['bb-torsions'] = bbtorsions_data.T
    
    sctorsions_feat = pyemma.coordinates.featurizer(pdb)
    sctorsions_feat.add_sidechain_torsions(cossin=True, periodic=False)
    sctorsions_data = pyemma.coordinates.load(xtc, features=sctorsions_feat)[start_frame:]
    feature_names['sc-torsions'] = sctorsions_feat
    features_data['sc-torsions'] = sctorsions_data.T

    bbdistances_feat = pyemma.coordinates.featurizer(pdb)
    bbdistances_feat.add_distances(bbdistances_feat.pairs(bbdistances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
    bbdistances_data = pyemma.coordinates.load(xtc, features=bbdistances_feat)[start_frame:]
    feature_names['bb-distances'] = bbdistances_feat
    features_data['bb-distances'] = bbdistances_data.T
    
    return feature_names, features_data
