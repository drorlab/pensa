import numpy as np
import glob
#from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *


# -- Functions to preprocess feature distributions for state clustering --

def import_distribution(simulation_folder, file_name):
    """
    Import data written to a txt file.

    Parameters
    ----------
    simulation_folder : str
        Path to the folder containing files for import.
    file_name : str
        Name of the file within the simulation_folder to be imported.

    Returns
    -------
    dists : array
        Array of the file data.

    """
    dists = np.loadtxt(simulation_folder + file_name,  delimiter=",")
    return dists

def get_filenames(folder):  
    """
    Obtain all the names of the files within a certain folder.

    Parameters
    ----------
    folder : str
        Location of the folder to obtain filenames.

    Returns
    -------
    files : list of str
        All filenames within folder.

    """
    files = [file.split(folder)[1] for file in glob.glob(folder + "*", recursive=True)]
    return files
    


def periodic_correction(angle1):
    """
    Correcting for the periodicity of angles [radians]. This ensures that
    states that span over the -pi, pi boundary are clustered as one state. 
    Waters featurized using PENSA are assigned dscrete state of 10000.0 for 
    representation of pocket occupancy, these values are ignored within the 
    periodic correction so that only the water orientation periodicity is handled.
    
    
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
    ##if the first bar height is not the minimum bar height
    ##then find the minimum bar and shift everything before that bar by 360
    if heights[0][0] > min(heights[0]):   
        ##define the new periodic boundary for the shifted values as the first minima
        perbound=heights[1][np.where(heights[0] == min(heights[0]))[0][0]+1]
        for angle_index in range(len(continuous_angles)):
            ##if the angle is before the periodic boundary, shift by 2*pi
            ## the boundaries in pyEMMA are in radians. [-pi, pi]
            if continuous_angles[angle_index] < perbound:
                continuous_angles[angle_index]+=2*np.pi
    for index in range(len(index_cont_angles)):
        new_dist[index_cont_angles[index]] = continuous_angles[index]
    
    
    
    return new_dist


    
def match_sim_lengths(sim1,sim2):
    """
    Make two lists the same length by truncating the longer list to match.
    This is necessary for clustering to ensure state clusters are not 
    biased towards one ensemble. Also this is necessary for SSI as a binary 
    distribution is created for the ensemble switch based off of equal length
    simulations.

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

