import numpy as np
import scipy as sp
import scipy.stats
import mdshare
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt




def write_cluster_traj(cluster_idx, top_file, trj_file, out_name, start_frame=0):
    """
    Writes a trajectory into a separate file for each cluster.
    
    Parameters
    ----------
        cluster_idx : int array
            Cluster index for each frame.
        top_file : str
            Reference topology for the second trajectory. 
        trj_file : str
            Trajetory file from which the frames are picked.
        out_name : str
            Core part of the name of the output files.
        start_frame : int, optional
            Frame from which to start reading the trajectory.
        
    """
    
    # Load and select the protein
    u = mda.Universe(top_file, trj_file)
    protein = u.select_atoms('all')
    print('Number of frames in trajectory:',len(u.trajectory))
    print('Number of cluster indices:',len(cluster_idx))
    return_protein = []
    # Loop over clusters
    num_clusters = np.max(cluster_idx)+1
    for nr in range(num_clusters):
        # For each cluster, write the corresponding frames to their new trajectory.
        with mda.Writer(out_name+"_c"+str(nr)+".xtc", protein.n_atoms) as W:
            for ts in u.trajectory:
                if ts.frame >= start_frame and cluster_idx[ts.frame-start_frame] == nr: 
                    W.write(protein)
                    return_protein.append(protein)
    return return_protein
                    

