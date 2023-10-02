import argparse
import numpy as np

# My own functions
from pensa import \
    get_structure_features, \
    obtain_combined_clusters, \
    write_cluster_traj, \
    wss_over_number_of_combined_clusters


# -------------#
# --- MAIN --- #
# -------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file_a", type=str,
                        default='traj/rhodopsin_arrbound_receptor.gro')
    parser.add_argument("--trj_file_a", type=str,
                        default='traj/rhodopsin_arrbound_receptor.xtc')
    parser.add_argument("--ref_file_b", type=str,
                        default='traj/rhodopsin_gibound_receptor.gro')
    parser.add_argument("--trj_file_b", type=str,
                        default='traj/rhodopsin_gibound_receptor.xtc')
    parser.add_argument("--label_a", type=str, default='Sim A')
    parser.add_argument("--label_b", type=str, default='Sim B')
    parser.add_argument("--out_plots", type=str,
                        default='plots/rhodopsin_receptor')
    parser.add_argument("--out_results", type=str,
                        default='results/rhodopsin_receptor')
    parser.add_argument("--out_frames_a", type=str,
                        default='clusters/rhodopsin_arrbound_receptor')
    parser.add_argument("--out_frames_b", type=str,
                        default='clusters/rhodopsin_gibound_receptor')
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--feature_type", type=str, default='bb-torsions')
    parser.add_argument("--algorithm", type=str, default='kmeans')
    parser.add_argument("--max_num_clusters", type=int, default=12)
    parser.add_argument("--write_num_clusters", type=int, default=2)
    parser.add_argument('--write', dest='write', action='store_true')
    parser.add_argument('--no-write', dest='write', action='store_false')
    parser.add_argument('--wss', dest='wss', action='store_true')
    parser.add_argument('--no-wss', dest='wss', action='store_false')
    parser.set_defaults(write=True, wss=True)
    args = parser.parse_args()

    # -- FEATURES --

    # Load Features
    feat_a, data_a = get_structure_features(
        args.ref_file_a, args.trj_file_a,
        args.start_frame, cossin=True
    )
    feat_b, data_b = get_structure_features(
        args.ref_file_b, args.trj_file_b,
        args.start_frame, cossin=True
    )
    # Report dimensions
    print('Feature dimensions from', args.trj_file_a)
    for k in data_a.keys():
        print(k, data_a[k].shape)
    print('Feature dimensions from', args.trj_file_b)
    for k in data_b.keys():
        print(k, data_b[k].shape)

    # -- CLUSTERING THE COMBINED DATA --

    ftype = args.feature_type

    # Calculate clusters from the combined data
    cc = obtain_combined_clusters(
        data_a[ftype], data_b[ftype],
        args.label_a, args.label_b,
        args.start_frame,
        args.algorithm, max_iter=100,
        num_clusters=args.write_num_clusters, min_dist=12,
        saveas=args.out_plots + '_combined-clusters_' + ftype + '.pdf'
    )
    cidx, cond, oidx, wss, centroids = cc

    # Write indices to results file
    np.savetxt(
        args.out_results + '_combined-cluster-indices.csv',
        np.array([cidx, cond, oidx], dtype=int).T,
        delimiter=', ', fmt='%i',
        header='Cluster, Condition, Index within condition'
    )

    # Write out frames for each cluster for each simulation
    if args.write:
        write_cluster_traj(
            cidx[cond == 0], args.ref_file_a, args.trj_file_a,
            args.out_frames_a, args.start_frame
        )
        write_cluster_traj(
            cidx[cond == 1], args.ref_file_b, args.trj_file_b,
            args.out_frames_b, args.start_frame
        )

    # -- Within-Sum-of-Squares (WSS) analysis --
    if args.wss:
        wss_avg, wss_std = wss_over_number_of_combined_clusters(
            data_a[ftype], data_b[ftype],
            label_a=args.label_a, label_b=args.label_b,
            start_frame=args.start_frame,
            algorithm=args.algorithm,
            max_iter=100, num_repeats=5,
            max_num_clusters=args.max_num_clusters,
            plot_file=args.out_plots + '_wss_' + ftype + '.pdf'
        )
