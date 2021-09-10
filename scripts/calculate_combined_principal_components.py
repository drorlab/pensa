import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.stats
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda

# My own functions
from pensa import *




# -------------#
# --- MAIN --- #
# -------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file_a",  type=str, default='traj/rhodopsin_arrbound_receptor.gro')
    parser.add_argument("--trj_file_a",  type=str, default='traj/rhodopsin_arrbound_receptor.xtc')
    parser.add_argument("--ref_file_b",  type=str, default='traj/rhodopsin_gibound_receptor.gro')
    parser.add_argument("--trj_file_b",  type=str, default='traj/rhodopsin_gibound_receptor.xtc')
    parser.add_argument("--out_plots",   type=str, default='plots/rhodopsin_receptor' )
    parser.add_argument("--out_pc",      type=str, default='pca/rhodopsin_receptor' )
    parser.add_argument("--out_results", type=str, default='results/rhodopsin_receptor' )
    parser.add_argument("--start_frame", type=int, default=0 )
    parser.add_argument("--feature_type", type=str, default='bb-torsions' )
    parser.add_argument("--num_eigenvalues", type=int, default=12 )
    parser.add_argument("--num_components",  type=int, default=3 )
    parser.add_argument("--feat_threshold",  type=float, default=0.4 )
    args = parser.parse_args()


    # -- FEATURES --

    # Load Features 
    feat_a, data_a = get_structure_features(args.ref_file_a, args.trj_file_a, args.start_frame, cossin=True)
    feat_b, data_b = get_structure_features(args.ref_file_b, args.trj_file_b, args.start_frame, cossin=True)
    # Report dimensions
    print('Feature dimensions from', args.trj_file_a)
    for k in data_a.keys(): 
        print(k, data_a[k].shape)
    print('Feature dimensions from', args.trj_file_b)
    for k in data_b.keys():            
        print(k, data_b[k].shape)


    # -- JOINT PCA --

    ftype = args.feature_type

    # Calculate the principal components of the combined data
    combined_data = np.concatenate([data_a[ftype], data_b[ftype]], 0)
    pca = calculate_pca(combined_data)
    # Plot the corresponding eigenvalues
    cn, ev = pca_eigenvalues_plot(pca, num=args.num_eigenvalues, 
                                  plot_file=args.out_plots+"_"+ftype+"_eigenvalues_combined.pdf")
    # Save them to a CSV file
    np.savetxt(args.out_results+"_"+ftype+"_eigenvalues_combined.csv", np.array([cn,ev]).T, 
               delimiter=',', header='Component, Eigenvalue')
    # Plot feature correlation with top components and print relevant features
    pca_features(pca, feat_a[ftype], 
                 args.num_components, args.feat_threshold,
                 plot_file=args.out_plots+"_"+ftype+"_feature_correlation.pdf")
    # Sort each of the trajectories along the top components of combined data
    sort_trajs_along_common_pc(data_a[ftype], data_b[ftype],
                               args.ref_file_a, args.ref_file_b, args.trj_file_a, args.trj_file_b,
                               args.out_pc, num_pc=args.num_components, start_frame=args.start_frame)
    # Plot histograms of both simulations along the common PCs
    compare_projections(data_a[ftype], data_b[ftype], pca,
                        num=args.num_components, saveas=args.out_plots+"_"+ftype+"_pc-comparison.pdf")



