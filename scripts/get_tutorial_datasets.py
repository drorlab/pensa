import os
import requests
import argparse
import MDAnalysis as mda
from pensa import download_from_gpcrmd


def subsample(psf_file, xtc_file, out_file):
    u = mda.Universe(psf_file,xtc_file)
    print(len(u.trajectory))
    a = u.select_atoms('all')
    with mda.Writer(out_file, a.n_atoms) as W:
        for ts in u.trajectory[400:500:10]:
            W.write(a)
    return


# -------------#
# --- MAIN --- #
# -------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", type=str, default='./mor-data')
    parser.add_argument('-s', "--subsample", action='store_true')
    args = parser.parse_args()

    # Create the directory
    root = args.directory
    os.makedirs(root, exist_ok=True)

    # MOR-apo
    psf_file = '11427_dyn_151.psf'
    pdb_file = '11426_dyn_151.pdb'
    download_from_gpcrmd(psf_file,root)
    download_from_gpcrmd(pdb_file,root)
    for sim in ['11423_trj_151','11424_trj_151','11425_trj_151']:
        xtc_file = sim+'.xtc'
        download_from_gpcrmd(xtc_file,root)
        if args.subsample:
            out_file = sim+'_subsampled.xtc'
            subsample(root+psf_file, root+xtc_file, root+out_file)

    # MOR-BU72
    psf_file = '11580_dyn_169.psf'
    pdb_file = '11579_dyn_169.pdb'
    download_from_gpcrmd(psf_file,root)
    download_from_gpcrmd(pdb_file,root)
    for sim in ['11576_trj_169','11577_trj_169','11578_trj_169']:
        xtc_file = sim+'.xtc'
        download_from_gpcrmd(xtc_file,root)
        if args.subsample:
            out_file = sim+'_subsampled.xtc'
            subsample(root+psf_file, root+xtc_file, root+out_file)

