import numpy as np
from pensa import *
import random 
import math



def get_txt_features_ala2(filename, num_frames, cossin=False):
    phi = []
    psi = []

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

            

