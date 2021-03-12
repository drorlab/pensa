import numpy as np
import glob
from pensa.features import *
from pensa.statesinfo import *


# -- Functions to preprocess feature distributions for state clustering --

def periodic_correction(angle1):
    """
    Correcting for the periodicity of angles [radians].  
    Waters featurized using PENSA and including discrete occupancy are handled.
    
    
    Parameters
    ----------
    angle1 : list
        Univariate distribution for a specific feature.

    Returns
    -------
    new_dist : list
        Periodically corrected distribution.

    """
    new_dist=angle1.copy()
    continuous_angles = [angle for angle in new_dist if angle != 10000.0]
    index_cont_angles = [index for index, angle in enumerate(new_dist) if angle != 10000.0]      
    heights=np.histogram(continuous_angles, bins=90, density=True)
    ## Shift everything before bin with minimum height by periodic amount
    if heights[0][0] > min(heights[0]):   
        perbound=heights[1][np.where(heights[0] == min(heights[0]))[0][0]+1]
        for angle_index in range(len(continuous_angles)):
            if continuous_angles[angle_index] < perbound:
                continuous_angles[angle_index]+=2*np.pi
    for index in range(len(index_cont_angles)):
        new_dist[index_cont_angles[index]] = continuous_angles[index]
    
    return new_dist


    
def match_sim_lengths(sim1,sim2):
    """
    Make two lists the same length by truncating the longer list to match.


    Parameters
    ----------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    Returns
    -------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    """
    if len(sim1)!=len(sim2):
        if len(sim1)>len(sim2):
            sim1=sim1[0:len(sim2)]
        if len(sim1)<len(sim2):
            sim2=sim2[0:len(sim1)]  
    return sim1, sim2

