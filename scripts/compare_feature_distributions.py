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
    parser.add_argument( "--ref_file_a",  type=str, default='traj/rhodopsin_arrbound_receptor.gro')
    parser.add_argument( "--trj_file_a",  type=str, default='traj/rhodopsin_arrbound_receptor.xtc')
    parser.add_argument( "--ref_file_b",  type=str, default='traj/rhodopsin_gibound_receptor.gro')
    parser.add_argument( "--trj_file_b",  type=str, default='traj/rhodopsin_gibound_receptor.xtc')
    parser.add_argument( "--out_plots",   type=str, default='plots/rhodopsin_receptor' )
    parser.add_argument( "--out_vispdb",  type=str, default='vispdb/rhodopsin_receptor' )
    parser.add_argument( "--out_results", type=str, default='results/rhodopsin_receptor' )
    parser.add_argument( "--start_frame", type=int, default=0 )
    parser.add_argument( "--print_num",   type=int, default=12 )
    args = parser.parse_args()


    # -- FEATURES --

    # Load Features 
    feat_a, data_a = get_features(args.ref_file_a, args.trj_file_a, args.start_frame)
    feat_b, data_b = get_features(args.ref_file_b, args.trj_file_b, args.start_frame)
    # Report dimensions
    print('Feature dimensions from', args.trj_file_a)
    for k in data_a.keys(): 
        print(k, data_a[k].shape)
    print('Feature dimensions from', args.trj_file_b)
    for k in data_b.keys():            
        print(k, data_b[k].shape)


    # -- BACKBONE TORSIONS --

    print('BACKBONE TORSIONS')

    # Relative Entropy analysis with BB torsions
    relen = relative_entropy_analysis(feat_a['bb-torsions'], feat_b['bb-torsions'], 
                                      data_a['bb-torsions'], data_b['bb-torsions'],
                                      bin_width=None, bin_num=10, verbose=False)
    names, jsd, kld_ab, kld_ba = relen

    # Save all results (per feature) in a CSV file 
    np.savetxt(args.out_results+'_bb-torsions_relative-entropy.csv', np.array(relen).T, 
               fmt='%s', delimiter=',', header='Name, JSD(A,B), KLD(A,B), KLD(B,A)')
    
    # Save the Jensen-Shannon distance as "B factor" in a PDB file
    vis = residue_visualization(names, jsd, args.ref_file_a, 
                                args.out_plots+"_bb-torsions_jsd.pdf", 
                                args.out_vispdb+"_bb-torsions_jsd.pdb",
                                y_label='max. JS dist. of BB torsions')

    # Save the per-residue data in a CSV file
    np.savetxt(args.out_results+'_bb-torsions_max-jsd-per-residue.csv', np.array(vis).T, 
               fmt='%s', delimiter=',', header='Residue, max. JSD(A,B)')

    # Print the features with the highest values
    print("Backbone torsions with the strongest deviations:")
    sf = sort_features(names, jsd)
    for f in sf[:args.print_num]: print(f[0], f[1])


    # -- SIDECHAIN TORSIONS --

    print('SIDECHAIN TORSIONS')

    # Relative Entropy analysis with sidechain torsions
    relen = relative_entropy_analysis(feat_a['sc-torsions'], feat_b['sc-torsions'],
                                      data_a['sc-torsions'], data_b['sc-torsions'],
                                      bin_width=None, bin_num=10, verbose=False)
    names, jsd, kld_ab, kld_ba = relen

    # Save all results (per feature) in a CSV file 
    np.savetxt(args.out_results+'_sc-torsions_relative-entropy.csv', np.array(relen).T,
               fmt='%s', delimiter=',', header='Name, JSD(A,B), KLD(A,B), KLD(B,A)') 

    # Save the Jensen-Shannon distance as "B factor" in a PDB file
    vis = residue_visualization(names, jsd, args.ref_file_a, 
                                args.out_plots+"_sc-torsions_jsd.pdf",
                                args.out_vispdb+"_sc-torsions_jsd.pdb",
                                y_label='max. JS dist. of SC torsions')

    # Save the per-residue data in a CSV file
    np.savetxt(args.out_results+'_sc-torsions_max-jsd-per-residue.csv', np.array(vis).T,
               fmt='%s', delimiter=',', header='Residue, max. JSD(A,B)')

    # Print the features with the highest values
    print("Sidechain torsions with the strongest deviations:")
    sf = sort_features(names, jsd)
    for f in sf[:args.print_num]: print(f[0], f[1])


    # -- BACKBONE C-ALPHA DISTANCES --

    print('BACKBONE C-ALPHA DISTANCES')

    # Relative entropy analysis for C-alpha distances
    relen = relative_entropy_analysis(feat_a['bb-distances'], feat_b['bb-distances'], 
                                      data_a['bb-distances'], data_b['bb-distances'],
                                      bin_width=0.01, verbose=False)
    names, jsd, kld_ab, kld_ba = relen 

    # Save all results (per feature) in a CSV file 
    np.savetxt(args.out_results+'_bb-distances_relative-entropy.csv', np.array(relen).T,
               fmt='%s', delimiter=',', header='Name, JSD(A,B), KLD(A,B), KLD(B,A)')

    # Print the features with the highest values
    print("Backbone C-alpha distances with the strongest deviations:")
    sf = sort_features(names, jsd)
    for f in sf[:args.print_num]: print(f[0], f[1])
    
    # Visualize the deviations in a matrix plot
    matrix = distances_visualization(names, jsd, 
                                     args.out_plots+"_bb-distances-distributions_jsd.pdf",
                                     vmin = 0.0, vmax = 1.0)

    # Difference-of-the-mean analysis for C-alpha distances
    meanda = mean_difference_analysis(feat_a['bb-distances'], feat_b['bb-distances'],
                                      data_a['bb-distances'], data_b['bb-distances'],
                                      verbose=False)
    names, avg, diff = meanda

    # Save all results (per feature) in a CSV file 
    np.savetxt(args.out_results+'_bb-distances_difference-of-mean.csv', np.array(meanda).T,
               fmt='%s', delimiter=',', header='Name, average, difference')

    # Sort the distances by their differences
    print("Backbone C-alpha distances with the strongest differences of their mean value:")
    sf = sort_features(names, diff)
    for f in sf[:args.print_num]: print(f[0], f[1])

    # Visualize the deviations in a matrix plot
    matrix = distances_visualization(names, diff, args.out_plots+"_bb-diststances_difference-of-mean.pdf")



