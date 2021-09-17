import os
import shutil
import subprocess
import tempfile
import mdtraj as md
import numpy as np
from diffnets.utils import get_fns

CURR_DIR = os.getcwd()
UP_DIR = CURR_DIR[:-len(CURR_DIR.split('/')[-1])]
CLI_DIR = UP_DIR + 'cli'

def test_preprocess_default_inds():
    curr_dir = os.getcwd()
    try:
        td = tempfile.mkdtemp(dir=curr_dir)
        ftmp = tempfile.NamedTemporaryFile(delete=False)
        traj_dirs_tmp = ftmp.name + ".npy"
        inp = np.array([os.path.join(curr_dir,"data/traj1"),
                    os.path.join(curr_dir,"data/traj2")])
        np.save(traj_dirs_tmp, inp, allow_pickle=False)

        ftmp2 = tempfile.NamedTemporaryFile(delete=False)
        pdb_fns_tmp = ftmp2.name + ".npy"
        inp = np.array([os.path.join(curr_dir,"data/beta-peptide1.pdb"),
                    os.path.join(curr_dir,"data/beta-peptide2.pdb")])
        np.save(pdb_fns_tmp, inp, allow_pickle=False)

        subprocess.call(['python', CLI_DIR + "/main.py", "process",
                        traj_dirs_tmp, pdb_fns_tmp, td])

        assert os.path.exists(os.path.join(td,"wm.npy"))
        assert os.path.exists(os.path.join(td,"uwm.npy"))
        assert os.path.exists(os.path.join(td,"master.pdb"))
        assert os.path.exists(os.path.join(td,"data"))

        xtc_fns = os.path.join(td,"aligned_xtcs")
        data_fns = get_fns(xtc_fns,"*.xtc")
        ind_fns = os.path.join(td,"indicators")
        inds = get_fns(ind_fns,"*.npy")

        print(len(data_fns))
        assert len(data_fns) == len(inds)

    finally:
        os.remove(traj_dirs_tmp)
        os.remove(pdb_fns_tmp)
        shutil.rmtree(td)

def test_preprocess_custom_inds():
    curr_dir = os.getcwd()
    try:
        td = tempfile.mkdtemp(dir=curr_dir) 
        ftmp = tempfile.NamedTemporaryFile(delete=False)
        traj_dirs_tmp = ftmp.name + ".npy"
        inp = np.array([os.path.join(curr_dir,"data/traj1"),
                    os.path.join(curr_dir,"data/traj2")])
        np.save(traj_dirs_tmp, inp, allow_pickle=False)
    
        ftmp2 = tempfile.NamedTemporaryFile(delete=False)
        pdb_fns_tmp = ftmp2.name + ".npy"
        inp = np.array([os.path.join(curr_dir,"data/beta-peptide1.pdb"),
                    os.path.join(curr_dir,"data/beta-peptide2.pdb")])
        np.save(pdb_fns_tmp, inp, allow_pickle=False) 

        ftmp3 = tempfile.NamedTemporaryFile(delete=False)
        inds_fn_tmp = ftmp3.name + ".npy"   
        pdb = md.load(inp[0])
        inds = pdb.top.select("name CA or name N or name CB or name C")
        both_inds = np.array([inds,inds])
        np.save(inds_fn_tmp, both_inds, allow_pickle=False)

        subprocess.call(['python', CLI_DIR + "/main.py", "process", 
                        traj_dirs_tmp, pdb_fns_tmp, td,
                        "-a" + inds_fn_tmp])

        assert os.path.exists(os.path.join(td,"wm.npy"))
        assert os.path.exists(os.path.join(td,"uwm.npy")) 
        assert os.path.exists(os.path.join(td,"master.pdb"))
        assert os.path.exists(os.path.join(td,"data"))    

        xtc_fns = os.path.join(td,"aligned_xtcs")
        data_fns = get_fns(xtc_fns,"*.xtc")
        ind_fns = os.path.join(td,"indicators")
        inds = get_fns(ind_fns,"*.npy")
        
        print(len(data_fns))
        assert len(data_fns) == len(inds)

    finally:
        os.remove(traj_dirs_tmp)
        os.remove(pdb_fns_tmp)
        os.remove(inds_fn_tmp)
        shutil.rmtree(td)

def test_train():
    curr_dir = os.getcwd()
    try:
        td = tempfile.mkdtemp(dir=curr_dir)
        ftmp = tempfile.NamedTemporaryFile(delete=False,mode="w+")

        params =["data_dir: '%s/data/whitened'" % curr_dir,
                 "n_epochs: 4",
                 "act_map: [0,1]",
                 "lr: 0.0001",
                 "n_latent: 10",
                 "hidden_layer_sizes: [50]",
                 "em_bounds: [[0.1,0.3],[0.6,0.9]]",
                 "do_em: True",
                 "em_batch_size: 50",
                 "nntype: 'nnutils.sae'",
                 "batch_size: 32",
                 "batch_output_freq: 50",
                 "epoch_output_freq: 2",
                 "test_batch_size: 50",
                 "frac_test: 0.1",
                 "subsample: 10",
                 "outdir: %s" % td,
                 "data_in_mem: False"
                ]
        
        for line in params:
            ftmp.write(line)
            ftmp.write("\n")
        ftmp.close()
        
        subprocess.call(['python', CLI_DIR + "/main.py", "train",
                        ftmp.name])

        assert os.path.exists(os.path.join(td,"nn_best_polish.pkl"))
    finally:
        os.remove(ftmp.name)
        shutil.rmtree(td)

def test_analyze():
    curr_dir = os.getcwd()
    try:
        subprocess.call(['python', CLI_DIR + "/main.py", "analyze",
                        "%s/data/whitened" % curr_dir,
                        "%s/data/trained_output" % curr_dir,
                         "-c", "20"])

        assert os.path.exists(os.path.join("%s/data/trained_output" % curr_dir,
                                            "rescorr-100.pml"))
        assert os.path.exists(os.path.join("%s/data/trained_output/rmsd.npy"
                                             % curr_dir))
        assert os.path.exists(os.path.join("%s/data/trained_output/labels"
                                             % curr_dir))
        assert os.path.exists(os.path.join("%s/data/trained_output/encodings"
                                             % curr_dir))
        assert os.path.exists(os.path.join("%s/data/trained_output/cluster_20"
                                             % curr_dir))
        assert os.path.exists(os.path.join("%s/data/trained_output/recon_trajs"
                                             % curr_dir))
        assert os.path.exists(os.path.join("%s/data/trained_output/morph_label"
                                             % curr_dir))

    finally:
        shutil.rmtree(os.path.join("%s/data/trained_output/encodings" % curr_dir))
        shutil.rmtree(os.path.join("%s/data/trained_output/labels" % curr_dir))
        shutil.rmtree(os.path.join("%s/data/trained_output/cluster_20" % curr_dir))
        shutil.rmtree(os.path.join("%s/data/trained_output/recon_trajs" % curr_dir))
        shutil.rmtree(os.path.join("%s/data/trained_output/morph_label" % curr_dir))
        os.remove(os.path.join("%s/data/trained_output/rmsd.npy" % curr_dir))
        os.remove(os.path.join("%s/data/trained_output/rescorr-100.pml" % curr_dir))
