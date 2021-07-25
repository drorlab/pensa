import os
import shutil
import mdtraj as md
import tempfile
import numpy as np
import torch

from diffnets.utils import get_fns
from diffnets.data_processing import ProcessTraj, WhitenTraj

CURR_DIR = os.getcwd()
UP_DIR = CURR_DIR[:-len(CURR_DIR.split('/')[-1])]
SCRIPTS_DIR = UP_DIR + 'scripts'

def assert_matrix_is_identity(M):
    print(np.max(np.abs(M - np.eye(M.shape[0]))))
    assert (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]), atol=1e-04)
    return True

def test_default_inds():

    try:
        td = tempfile.mkdtemp(dir=CURR_DIR)
        data_dir = os.path.join(CURR_DIR,"data")
        pdb_fn1 = os.path.join(data_dir, "beta-peptide1.pdb")
        pdb = md.load(pdb_fn1)
        inds1 = pdb.top.select("name CA or name C or name CB or name N")
        pdb_fn2 = os.path.join(data_dir, "beta-peptide2.pdb")
        pdb = md.load(pdb_fn2)
        inds2 = pdb.top.select("name CA or name C or name CB or name N")

        var_dir_names = [os.path.join(data_dir,"traj1"),
                    os.path.join(data_dir,"traj2")]
        proc_traj = ProcessTraj(var_dir_names,[pdb_fn1,pdb_fn2],td)
        assert set(proc_traj.atom_sel[0]) == set(inds1)
        assert set(proc_traj.atom_sel[1]) == set(inds2) 

    finally:
        shutil.rmtree(td)

def test_whitening_correctness1():

    w = WhitenTraj("./data/whitened/")
    master = md.load("./data/whitened/master.pdb")
    traj1 = md.load("./data/whitened/aligned_xtcs/000000.xtc",top=master)
    traj2 = md.load("./data/whitened/aligned_xtcs/000001.xtc",top=master)
    wm = np.load("./data/whitened/wm.npy")
    w.apply_whitening_xtc_dir(w.xtc_dir,master.top,wm,
                              w.cm,1,"./data/whitened/whitened_xtcs")
    traj_fns = get_fns("./data/whitened/whitened_xtcs/", "*.xtc")
    traj = md.load(traj_fns[0],top=master)
    coords = traj.xyz.reshape((2501, 3 * 39))
    c00_1 = np.matmul(coords.transpose(),coords)

    traj = md.load(traj_fns[1],top=master)
    coords = traj.xyz.reshape((2500, 3 * 39))
    c00_2 = np.matmul(coords.transpose(),coords)
    c00 = c00_1 + c00_2
    c00 /= 5001

    # assert_matrix_is_identity(c00)
    assert (np.abs(117 - np.sum(np.diagonal(c00))) < 1)

def test_whitening_correctness2():
    w = WhitenTraj("./data/whitened")
    # generate dummy data
    X = np.random.rand(100, 30)
    X_s = X - X.mean(axis=0)
    cov = np.cov(X_s.transpose())
    # get whitening matrix
    uwm, wm = w.get_wuw_mats(cov)

    Y = w.apply_whitening(X_s, wm, X_s.mean(axis=0))
    whitened_cov = np.cov(Y.transpose())
    # assert that covariance of whitened data is identity
    assert_matrix_is_identity(whitened_cov)
    assert np.abs(np.sum(whitened_cov) - Y.shape[1]) < .0001
    assert np.abs(np.sum(np.diagonal(whitened_cov)) - Y.shape[1]) < .0001

def test_whitening_correctness3():
    '''
    This tests assess if whitening matrix can
    be correctly generated for a real covariance matrix
    '''
    from scipy.linalg import inv, sqrtm

    w = WhitenTraj("data/whitened")

    # Load three different covariance matrices
    # cov1 is a DiffNets-generated matrix for 8 myosin isoforms with a subset of atoms relative to cov2
    # cov2 is a DiffNets-generated matrix for 8 myosin isoforms
    # cov3 is beta-lactamase covariance matrix
    for i in range(1, 4):
        # Load covariance matrix
        cov = np.load(f"data/cov/cov{i}.npy")

        e, v = torch.symeig(torch.from_numpy(cov).double(), eigenvectors=True)
        # In valid covariance matrix the smallest eigenvalue should be positive
        # because the covariance matrix is a positive semidefinite matrix
        # https://stats.stackexchange.com/questions/52976/is-a-sample-covariance-matrix-always-symmetric-and-positive-definite
        assert torch.min(e) > 0.0

        # Covariance matrix should be symmetric
        assert np.allclose(cov, cov.T)

        uwm, wm = w.get_wuw_mats(cov.astype(np.double))

        # ZCA whitening matrix should be symmteric
        assert np.allclose(wm, wm.T)

        # assert that whitening and whitening produce the identity matrix
        assert_matrix_is_identity(np.matmul(wm, uwm))

        # assert that the covariance matrix multiplied by the transpose of
        # the whitening matrix multiplied by the whitening matrix is identity
        # for detailed explanation see page 3 of https://arxiv.org/pdf/1512.00809.pdf
        test = np.matmul(cov, np.matmul(wm.transpose(), wm))
        assert_matrix_is_identity(test)


