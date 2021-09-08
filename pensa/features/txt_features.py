import numpy as np
from pensa import *
import random 
import math



def get_txt_features_ala2(filename, num_frames, cossin=False):
    """
    Parses features for ala2 from a text file. 
    The text file must be formatted with "phi", followed by all phi angles, a blank line,
    followed by "psi" and all psi angles, with one angle per line.

    Parameters
    ----------
    filename : str
        File name of the text file.
    num_frames : int
        Maximum number of trajectory frames used in features array.
    cossin : bool
        Determines if the features array contains torsion angles or the sin and cos of torsion angles.

    Returns
    -------
        features : numpy array
            Data for all features

    """
    phi = []
    psi = []
    curr = 'phi'
    with open(filename) as f:
        for s in f.readlines():
            if s == 'phi\n' or s == 'psi\n':
                continue
            if s == '\n':
                curr = 'psi'
            else:
                if curr == 'phi':
                    phi.append(float(s))
                else:
                    psi.append(float(s))

    if len(phi) > num_frames:
        temp = list(zip(phi, psi))
        random.shuffle(temp)
        phi, psi = zip(*temp)

    features = []
    if not cossin:
        features = np.zeros((num_frames, 2))
        for i in range(num_frames):
            features[i, 0] = phi[i]
            reatures[i, 1] = psi[i]
    else:
        features = np.zeros((num_frames, 4))
        for i in range(num_frames):
            features[i, 0] = math.cos(phi[i])
            features[i, 1] = math.sin(phi[i])
            features[i, 2] = math.cos(psi[i])
            features[i, 3] = math.sin(psi[i])

    return features

            

