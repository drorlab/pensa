import mdtraj as md
import numpy as np
import itertools
from . import utils
import multiprocessing as mp
import os
import functools
from torch.autograd import Variable
import torch
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import enspara
import enspara.cluster as cluster
import enspara.info_theory as infotheor
import enspara.msm as msm
import enspara.cluster as cluster
import enspara.info_theory as infotheor
import pickle
import scipy.sparse
import sys
from pylab import *
from torch.autograd import Variable
from collections import defaultdict

class Analysis:
    """Core object for running analysis.

    Parameters
    ----------
    net : nnutils object
        Neural network to perform analysis with
    netdir : str
        path to directory with neural network results
    datadir : str
        path to directory with data required to train the data. Includes
        cm.npy, wm.npy, uwm.npy, master.pdb, an aligned_xtcs dir, and an
        indicators dir.
    """
    def __init__(self, net, netdir, datadir):
        self.net = net
        self.netdir = netdir
        self.datadir = datadir
        self.top = md.load(os.path.join(
                           self.datadir, "master.pdb"))
        self.cm = np.load(os.path.join(self.datadir, "cm.npy"))
        self.n_cores = mp.cpu_count()

    def encode_data(self):
        """Calculate the latent space for all trajectory frames.
        """
        enc_dir = os.path.join(self.netdir, "encodings")
        utils.mkdir(enc_dir)
        xtc_dir = os.path.join(self.datadir, "aligned_xtcs")
        encode_dir(self.net, xtc_dir, enc_dir, self.top, self.n_cores, self.cm)

    def recon_traj(self):
        """Reconstruct all trajectory frames using the trained neural 
        network"""
        recon_dir = os.path.join(self.netdir, "recon_trajs")
        utils.mkdir(recon_dir)
        enc_dir = os.path.join(self.netdir, "encodings")
        recon_traj_dir(self.net, enc_dir, recon_dir, self.top.top,
                       self.cm, self.n_cores)
        print("trajectories reconstructed")

    def get_labels(self):
        """Calculate the classification score for all trajectory frames
        """
        label_dir = os.path.join(self.netdir, "labels")
        utils.mkdir(label_dir)
        enc_dir = os.path.join(self.netdir, "encodings")
        calc_labels(self.net, enc_dir, label_dir, self.n_cores)
        print("labels calculated for all states")

    def get_rmsd(self):
        """Calculate RMSD between actual trajectory frames and autoencoder
        reconstructed frames"""
        rmsd_fn = os.path.join(self.netdir, "rmsd.npy")
        recon_dir = os.path.join(self.netdir, "recon_trajs")
        orig_xtc_dir = os.path.join(self.datadir, "aligned_xtcs")
        rmsd = rmsd_dists_dir(recon_dir, orig_xtc_dir, self.top, self.n_cores)
        np.save(rmsd_fn, rmsd)

    def morph(self,n_frames=10):
        """Get representative structures for classification scores
        from 0 to 1.

        Parameters
        ----------
        n_frames : int
            How many representative structures to output. Bins between
            0 and 1 will be calculated with this number.
        """
        morph_label(self.net,self.netdir,self.datadir,n_frames=n_frames)

    def assign_labels_to_variants(self,plot_labels=False):
        """Map DiffNet labels to each variant with option to plot
           a histogram of the labels.

        Parameters
        ----------
        plot_labels : optional, boolean
            Save a matplotlob figure of the label histogram.

        Returns
        -------
        lab_v : dictionary
            Dictionary mapping labels to their respective variants.
        """

        lab_fns = utils.get_fns(os.path.join(self.netdir,"labels"),"*.npy")
        traj_d_path = os.path.join(self.datadir,"traj_dict.pkl")
        traj_d = pickle.load(open(traj_d_path, 'rb')) 
        lab_v = defaultdict(list)
        for key,item in traj_d.items():
            for traj_ind in range(item[0],item[1]):
                lab = np.load(lab_fns[traj_ind])
                lab_v[key].append(lab) 

        if plot_labels:
            plt.figure(figsize=(16,16))
            axes = plt.gca()
            lw = 8

            for k in traj_d.keys():
                t = np.concatenate(lab_v[k])
                n, x = np.histogram(t, range=(0, 1), bins=50)
                plt.plot(x[:-1],n,label=k,linewidth=lw)


            plt.xticks(fontsize=36)
            plt.yticks(fontsize=36)
            axes.set_xlabel('DiffNet Label',labelpad=40, fontsize=36)
            axes.set_ylabel('# of Simulation Frames',labelpad=40,fontsize=36)
            axes.tick_params(direction='out', length=20, width=5,
                           grid_color='r', grid_alpha=0.5)
            plt.legend(fontsize=36)

            for axis in ['top','bottom','left','right']:
                axes.spines[axis].set_linewidth(5)

            plt.savefig(os.path.join(self.netdir,"label_plot.png"))

        return lab_v

    def find_feats(self,inds,out_fn,n_states=2000,num2plot=100,clusters=None):
        """Generate a .pml file that will show the distances that change
        in a way that is most with changes in the classifications score.

        Parameters
        ----------
        inds : np.ndarray,
            Indices of the topology file that are to be included in
            calculating what distances are most correlated with classification
            score.
        out_fn : str
            Name of the output file.
        n_states : int (default=2000)
            How many cluster centers to calculate and use for correlation
            measurement.
        num2plot : int (default=100)
            Number of distances to be shown.
        clusters : enspara cluster object
            Cluster object with center_indices attribute
        """
        if not clusters:
            cc_dir = os.path.join(self.netdir, "cluster_%d" % n_states)
            utils.mkdir(cc_dir)

            enc = utils.load_npy_dir(os.path.join(self.netdir, "encodings"), "*npy")
            if hasattr(self.net,"split_inds"):
                x = self.net.encoder1[-1].out_features
                enc = enc[:,:x]
            clusters = cluster.hybrid.hybrid(enc, euc_dist,
                                         n_clusters=n_states, n_iters=1)
            cluster_fn = os.path.join(cc_dir, "clusters.pkl")
            pickle.dump(clusters, open(cluster_fn, 'wb'))

        find_features(self.net,self.datadir,self.netdir,
            clusters.center_indices,inds,out_fn,num2plot=num2plot)

    def run_core(self):
        """Wrapper to run the analysis functions that should be 
        run after training.
        """
        self.encode_data()
        
        self.recon_traj()

        self.get_labels()
        
        self.get_rmsd()

def euc_dist(trj, frame):
    diff = np.abs(trj - frame)
    try:
        d = np.sqrt(np.sum(diff * diff, axis=1))
    except:
        d = np.array([np.sqrt(np.sum(diff * diff))])
    return d

def recon_traj(enc, net, top, cm):
    n = len(enc)
    n_atoms = top.n_atoms
    x = Variable(torch.from_numpy(enc).type(torch.FloatTensor))
    coords = net.decode(x)
    coords = coords.detach().numpy()
    coords += cm
    coords = coords.reshape((n, n_atoms, 3))
    traj = md.Trajectory(coords, top)
    return traj

def _recon_traj_dir(enc_fn, net, recon_dir, top, cm):
    enc = np.load(enc_fn)
    traj = recon_traj(enc, net, top, cm)

    new_fn = os.path.split(enc_fn)[1]
    base_fn = os.path.splitext(new_fn)[0]
    new_fn = base_fn + ".xtc"
    new_fn = os.path.join(recon_dir, new_fn)
    traj.save(new_fn)

def recon_traj_dir(net, enc_dir, recon_dir, top, cm, n_cores):
    enc_fns = utils.get_fns(enc_dir, "*.npy")
    
    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_recon_traj_dir, net=net, recon_dir=recon_dir, top=top, cm=cm)
    pool.map(f, enc_fns)
    pool.close()

def _calc_labels(enc_fn, net, label_dir):
    enc = np.load(enc_fn)
    if hasattr(net,"split_inds"):
        x = net.encoder1[-1].out_features
        enc = enc[:,:x]
    enc = Variable(torch.from_numpy(enc).type(torch.FloatTensor))
    labels = net.classify(enc)
    labels = labels.detach().numpy()

    new_fn = os.path.split(enc_fn)[1]
    new_fn = os.path.join(label_dir, "lab" + new_fn)
    np.save(new_fn, labels)

def calc_labels(net, enc_dir, label_dir, n_cores):
    enc_fns = utils.get_fns(enc_dir, "*npy")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_calc_labels, net=net, label_dir=label_dir)
    pool.map(f, enc_fns)
    pool.close()

def get_rmsd_dists(orig_traj, recon_traj):
    n_frames = len(recon_traj)
    if n_frames != len(orig_traj):
        # should raise exception
        print("Can't get rmsds between trajectories of different lengths")
        return
    pairwise_rmsd = []
    for i in range(0, n_frames, 10):
        r = md.rmsd(recon_traj[i], orig_traj[i], parallel=False)[0]
        pairwise_rmsd.append(r)
    pairwise_rmsd = np.array(pairwise_rmsd)
    return pairwise_rmsd

def _rmsd_dists_dir(recon_fn, orig_xtc_dir, ref_pdb):
    recon_traj = md.load(recon_fn, top=ref_pdb.top)
    base_fn = os.path.split(recon_fn)[1]
    orig_fn = os.path.join(orig_xtc_dir, base_fn)
    orig_traj = md.load(orig_fn, top=ref_pdb.top)
    pairwise_rmsd = get_rmsd_dists(orig_traj, recon_traj)
    return pairwise_rmsd

def rmsd_dists_dir(recon_dir, orig_xtc_dir, ref_pdb, n_cores):
    recon_fns = utils.get_fns(recon_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_rmsd_dists_dir, orig_xtc_dir=orig_xtc_dir, ref_pdb=ref_pdb)
    res = pool.map(f, recon_fns)
    pool.close()

    pairwise_rmsd = np.concatenate(res)
    return pairwise_rmsd

def _encode_dir(xtc_fn, net, outdir, top, cm):
    traj = md.load(xtc_fn, top=top)
    n = len(traj)
    n_atoms = traj.top.n_atoms
    x = traj.xyz.reshape((n, 3*n_atoms))-cm
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
    if hasattr(net, 'split_inds'):
        lat1, lat2 = net.encode(x)
        output = torch.cat((lat1,lat2),1)
    else:
        output = net.encode(x)
    output = output.detach().numpy()
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + ".npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, output)

def encode_dir(net, xtc_dir, outdir, top, n_cores, cm):
    xtc_fns = utils.get_fns(xtc_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_encode_dir, net=net, outdir=outdir, top=top, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()

def morph_label(net,nn_dir,data_dir,n_frames=10):
    pdb_fn = os.path.join(data_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    uwm = np.load(uwm_fn)
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    enc = utils.load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    n_latent = int(enc.shape[1])
    morph_dir = os.path.join(nn_dir, "morph_label")
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    labels_dir = os.path.join(nn_dir,"labels")
    labels = utils.load_npy_dir(labels_dir,"*.npy")
    labels = labels.flatten()

    my_min = np.min(labels)
    my_max = np.max(labels)
    morph_enc = np.zeros((n_frames,n_latent))
    vals = np.linspace(my_min, my_max, n_frames)
    delta = (vals[1] - vals[0]) * 0.5
    for i in range(n_frames):
        val = vals[i]
        inds = np.where(np.logical_and(labels>=val-delta,
                                   labels<=val+delta))[0]

        for j in range(n_latent):
            x = np.mean(enc[inds,j])
            morph_enc[i, j] = x

    #morph_enc = Variable(torch.from_numpy(morph_enc).type(torch.FloatTensor))
    morph_enc = np.array(morph_enc)
    traj = recon_traj(morph_enc,net,ref_s.top,cm)
    rmsf = get_rmsf(traj)

    out_fn = os.path.join(morph_dir, "morph_0-1.pdb")
    traj.save_pdb(out_fn, bfactors=rmsf)

def get_rmsf(traj):
    x_mean = traj.xyz.mean(axis=0)
    delta = traj.xyz - x_mean
    d2 = np.einsum('ijk,ijk->ij', delta, delta)
    p = 1.0*np.ones(len(traj)) / len(traj)
    msf = np.einsum('ij,i->j', d2, p)
    return np.sqrt(msf)

def find_features(net,data_dir,nn_dir,clust_cents,inds,out_fn,num2plot=100):
    #Need to atom custom indices
    encs_dir = os.path.join(nn_dir,"encodings")
    encs = utils.load_npy_dir(encs_dir,"*.npy")
    encs = encs[clust_cents]

    cm = np.load(os.path.join(data_dir,"cm.npy"))
    top = md.load(os.path.join(data_dir,"master.pdb"))
    traj = recon_traj(encs,net,top.top,cm)
    print("trajectory calculated")
    all_pairs = list(itertools.product(inds, repeat=2))
    distances = md.compute_distances(traj,all_pairs)

    labels_dir = os.path.join(nn_dir,"labels")
    labels = utils.load_npy_dir(labels_dir,"*.npy")
    labels = labels[clust_cents]

    n = len(inds)
    print(n, " distances being calculated")
    r_values = []
    slopes = []
    for i in np.arange(n*n):
        slope, intercept, r_value, p_value, std_err = stats.linregress(labels.flatten(),distances[:,i])
        r_values.append(r_value)
        slopes.append(slope)

    r2_values = np.array(r_values)**2
    corr_slopes = []
    count = 0
    print("Starting to write pymol file")
    f = open(os.path.join(nn_dir,out_fn), "w")
    for i in np.argsort(r2_values)[-num2plot:]:
        corr_slopes.append(slopes[i])
        #print(slopes[i],r2_values[i],i)
        j,k = np.array(all_pairs)[i,:]
        jnum = top.top.atom(j).residue.resSeq
        jname = top.top.atom(j).name
        knum = top.top.atom(k).residue.resSeq
        kname = top.top.atom(k).name
        if slopes[i] < 0:
            f.write("distance dc%s, master and resi %s and name %s, master and resi %s and name %s\n" % (count,jnum,jname,knum,kname))
            f.write("color red, dc%s\n" % count)
            f.write("hide label\n")
        else:
            f.write("distance df%s, master and resi %s and name %s, master and resi %s and name %s\n" % (count,jnum,jname,knum,kname))
            f.write("color blue, df%s\n" % count)
            f.write("hide label\n")
        count+=1
    f.close()

#########################################################
#                                                       #
#                Extra analysis functions               #       
#                                                       #
#########################################################


def calc_auc(net_fn,out_fn,data,labels):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    full_x = torch.from_numpy(data).type(torch.FloatTensor)
    if hasattr(net, "encode"):
        full_x = Variable(full_x.view(-1, 784).float())
        pred_x, latents, pred_class = net(full_x)
        preds = pred_class.detach().numpy()
    else:
        full_x = Variable(full_x.view(-1, 3,32,32).float())
        preds = net(full_x).detach().numpy()
    fpr, tpr, thresh = roc_curve(labels,preds)
    auc = roc_auc_score(labels,preds.flatten())
    print("AUC: %f" % auc)
    #plt.figure()
    #lw = 2
    #plt.plot(fpr, tpr, color='darkorange',
    #     lw=lw, label='ROC curve (area = %f)' % auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.savefig(out_fn)
    #plt.close()
    return auc, fpr, tpr

def split_vars(d, vars):
    n = len(d)
    n_vars = len(vars)
    n_per_var = int(len(d)/n_vars)
    lst = {}
    for i in range(n_vars):
        v = vars[i]
        lst[v] = d[i*n_per_var:(i+1)*n_per_var]
    return lst


def get_extrema(lst_lst):
    my_min = np.inf
    my_max = -np.inf
    for lst in lst_lst:
        my_min = np.min((my_min, np.min(lst)))
        my_max = np.max((my_max, np.max(lst)))
    return my_min, my_max


def common_hist(lst_lst, labels, bins):
    my_min, my_max = get_extrema(lst_lst)
    n_lst = len(lst_lst)
    all_h = {}
    for i in range(n_lst):
        h, x = np.histogram(lst_lst[i], bins=bins, range=(my_min, my_max))
        all_h[labels[i]] = h
    return all_h, x

def calc_overlap(d1, d2, bins):
    n_feat = d1.shape[1]
    js = np.zeros(n_feat)
    ent1 = np.zeros(n_feat)
    ent2 = np.zeros(n_feat)
    for i in range(n_feat):
        h, x = common_hist([d1[:, i], d2[:, i]], ["d1", "d2"], bins)
        h1 = h["d1"]
        h2 = h["d2"]
        p1 = np.array(h1) / h1.sum()
        p2 = np.array(h2) / h2.sum()
        js[i] = infotheor.js_divergence(p1, p2)
        ent1[i] = infotheor.shannon_entropy(p1)
        ent2[i] = infotheor.shannon_entropy(p2)
    return js, ent1, ent2

def project(enc, lab, vars, i1, i2, bins, my_title, cutoff=0.8):
    subsample = 100

    all_act_inds = np.where(lab>cutoff)[0]
    act_i1_mu = enc[all_act_inds, i1].mean()
    act_i1_std = enc[all_act_inds, i1].std()
    act_i2_mu = enc[all_act_inds, i2].mean()
    act_i2_std = enc[all_act_inds, i2].std()

    n_vars = len(vars)
    enc_dict = split_vars(enc, vars)
    lab_dict = split_vars(lab, vars)
    i1_dict = {}
    i2_dict = {}
    act_inds = {}
    for v in vars:
        i1_dict[v] = enc_dict[v][:, i1]
        i2_dict[v] = enc_dict[v][:, i2]
        act_inds[v] = np.where(lab_dict[v]>cutoff)[0]
    # i1_min, i1_max = get_extrema(i1_dict.values())
    # i2_min, i2_max = get_extrema(i2_dict.values())

    # drop outliers by only show data within const*std
    const = 3
    i1_mu = enc[:, i1].mean()
    i1_std = enc[:, i1].std()
    i1_min = np.max((i1_mu-const*i1_std, enc[:, i1].min()))
    i1_max = np.min((i1_mu+const*i1_std, enc[:, i1].max()))
    i2_mu = enc[:, i2].mean()
    i2_std = enc[:, i2].std()
    i2_min = np.max((i2_mu-const*i2_std, enc[:, i2].min()))
    i2_max = np.min((i2_mu+const*i2_std, enc[:, i2].max()))

    # get min/max of z dim
    cmin = np.inf
    cmax = -np.inf
    for i in range(n_vars):
        v = vars[i]
        tmp, x, y = np.histogram2d(i1_dict[v], i2_dict[v], range=([i1_min, i1_max], [i2_min, i2_max]), bins=n_bins)
        tmp /= tmp.sum()
        h = np.zeros(tmp.shape)
        inds = np.where(tmp>0)
        h[inds] = -np.log(tmp[inds])
        #inds = np.where(np.isnan(h))
        #h[inds] = 0
        cmin = np.min((cmin, h[inds].min()))
        cmax = np.max((cmax, h[inds].max()))

    height = 4
    width = height*n_vars
    fig = figure(figsize=(width, height))
    fig.suptitle(my_title)
    bins = 20
    dot_size = 0.1
    for i in range(n_vars):
        v = vars[i]
        ax = fig.add_subplot(1, n_vars, i+1, aspect='auto', xlim=x[[0, -1]], ylim=y[[0, -1]])
        #scatter(i1_dict[v], i2_dict[v], s=dot_size, c='b', alpha=0.1)
        tmp, x, y = np.histogram2d(i1_dict[v], i2_dict[v], range=([i1_min, i1_max], [i2_min, i2_max]), bins=n_bins)
        tmp /= tmp.sum()
        h = cmax*np.ones(tmp.shape)
        inds = np.where(tmp>0)
        h[inds] = -np.log(tmp[inds])
        h -= cmax
        delta_x = (x[1]-x[0])/2.0
        delta_y = (y[1]-y[0])/2.0
        #imshow(h, interpolation='bilinear', aspect='auto', origin='low', extent=[x[0]+delta_x, x[-1]+delta_x, y[0]+delta_y, y[-1]+delta_y], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        # transpose to put first dimension (i1) on x axis
        #imshow(h.T, interpolation='bilinear', aspect='auto', origin='low', extent=[y[0]+delta_y, y[-1]+delta_y, x[0]+delta_x, x[-1]+delta_x], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        imshow(h.T, interpolation='bilinear', aspect='auto', origin='low', extent=[x[0]+delta_x, x[-1]+delta_x, y[0]+delta_y, y[-1]+delta_y], vmin=cmin-cmax, vmax=0, cmap=get_cmap('Blues_r'))
        colorbar()
        
        lines = []
        line_labels = []
        for v2 in vars:
            i1_mu = i1_dict[v2].mean()
            i1_std = i1_dict[v2].std()
            i2_mu = i2_dict[v2].mean()
            i2_std = i2_dict[v2].std()
            #print(v, "x", i1_mu, i1_std)
            #print(v, "y", i2_mu, i2_std)
            line, _, _ = errorbar([i1_mu], [i2_mu], xerr=[i1_std], yerr=[i2_std], label=v2)
            lines.append(line)
            line_labels.append(v2)

            # inds = act_inds[v2]
            # if inds.shape[0] > subsample:
            #     inds = inds[::subsample]
            # print(inds.shape)
            # if inds.shape[0] > 0:
            #     scatter(i1_dict[v2][inds], i2_dict[v2][inds], s=dot_size, c='k')

        line, _, _ = errorbar([act_i1_mu], [act_i2_mu], xerr=[act_i1_std], yerr=[act_i2_std], label='act', ecolor='k', fmt='k')
        lines.append(line)
        line_labels.append('act')
        #legend()

        title(v)
    # scatter([0], [0], s=dot_size*10, c='k')
    # scatter([6], [0], s=dot_size*10, c='k')
    # scatter([6], [6], s=dot_size*10, c='k')
    fig.legend(lines, line_labels)
    show()

def morph_conditional(nn_dir, data_dir, n_frames=10):
    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    uwm = np.load(uwm_fn)
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    enc = load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    n_latent = int(enc.shape[1])
    morph_dir = os.path.join(nn_dir, "morph")
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    for i in range(n_latent):
        my_min, my_max = get_extrema([enc[:, i]])
        print(i, my_min, my_max)
        morph_enc = np.zeros((n_frames, n_latent))
        vals = np.linspace(my_min, my_max, n_frames)
        delta = (vals[1] - vals[0]) * 0.5
        for j in range(n_frames):
            val = vals[j]

            # set each latent variable to most probable value given latent(ind) within delta of selected value
            inds = np.where(np.logical_and(enc[:,i]>=val-delta, enc[:,i]<=val+delta))[0]
            for k in range(n_latent):
                n, x = np.histogram(enc[inds, k], bins=20)
                offset = (x[1] - x[0]) * 0.5
                morph_enc[j, k] = x[n.argmax()] + offset

            # fix ref latent variable to val
            morph_enc[j, i] = val

        morph_enc = Variable(torch.from_numpy(morph_enc).type(torch.FloatTensor))
        try:
            outputs, labs = net.decode(morph_enc)
        except:
            print("single")
            outputs = net.decode(morph_enc)
        outputs = outputs.data.numpy()
        coords = whiten.apply_unwhitening(outputs, uwm, cm)
        print("shape", coords.shape)
        recon_trj = md.Trajectory(coords.reshape((n_frames, n_atoms, 3)), ref_s.top)
        out_fn = os.path.join(morph_dir, "m%d.pdb" % i)
        recon_trj.save(out_fn)

def morph_cond_mean(nn_dir,data_dir,n_frames=10):
    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    uwm = np.load(uwm_fn)
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)
    enc = load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    n_latent = int(enc.shape[1])
    morph_dir = os.path.join(nn_dir, "morph_bin_mean")
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    for i in range(n_latent):
        my_min, my_max = get_extrema([enc[:, i]])
        print(i, my_min, my_max)
        morph_enc = np.zeros((n_frames, n_latent))
        vals = np.linspace(my_min, my_max, n_frames)
        delta = (vals[1] - vals[0]) * 0.5
        for j in range(n_frames):
            val = vals[j]

            # set each latent variable to most probable value given latent(ind) within delta of selected value
            inds = np.where(np.logical_and(enc[:,i]>=val-delta, enc[:,i]<=val+delta))[0]
            for k in range(n_latent):
                x  = np.mean(enc[inds,k])
                morph_enc[j, k] = x

            # fix ref latent variable to val
            morph_enc[j, i] = val

        morph_enc = Variable(torch.from_numpy(morph_enc).type(torch.FloatTensor))
        traj = utils.recon_traj(morph_enc,net,ref_s.top,cm)
        rmsf = get_rmsf(traj)

        out_fn = os.path.join(outdir, "m%d.pdb" % i)
        traj.save_pdb(out_fn, bfactors=rmsf)

def morph_std(nn_dir, data_dir, enc):
    outdir = os.path.join(nn_dir, "morph_std")
    utils.mkdir(outdir)
    n_frames = 10

    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    n_atoms = ref_s.top.n_atoms
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)

    n_latent = int(enc.shape[1])
    ave_enc = enc.mean(axis=0)
    std_enc = enc.std(axis=0)
    max_enc = enc.max(axis=0)
    min_enc = enc.min(axis=0)

    # want vary between mean +/- 2*std but not go out of range
    for i in range(n_latent):
        #my_min = np.max((ave_enc[i]-5*std_enc[i], min_enc[i]))
        #my_max = np.min((ave_enc[i]+5*std_enc[i], max_enc[i]))
        my_min = min_enc[i]
        my_max = max_enc[i]

        morph_enc = np.zeros((n_frames, n_latent)) + ave_enc
        morph_enc[:, i] = np.linspace(my_min, my_max, n_frames)
        traj = utils.recon_traj(morph_enc, net, ref_s.top, cm)

        rmsf = get_rmsf(traj)

        out_fn = os.path.join(outdir, "m%d.pdb" % i)
        traj.save_pdb(out_fn, bfactors=rmsf)

def get_act_inact(nn_dir, data_dir, enc, labels):
    """Save most active/inactive sturctures with RMSDs from target less than 2 Angstroms."""
    outdir = os.path.join(nn_dir, "act_and_inact")
    utils.mkdir(outdir)
    n_extreme = 1000

    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_dir, 'rb'))
    net.cpu()
    pdb_fn = os.path.join(nn_dir, "master.pdb")
    ref_s = md.load(pdb_fn)
    ca_inds = ref_s.top.select('name CA')
    n_atoms = ref_s.top.n_atoms
    cm_fn = os.path.join(data_dir, "cm.npy")
    cm = np.load(cm_fn)

    rmsd_cutoff = 0.2
    rmsd_fn = os.path.join(nn_dir, "rmsd.npy")
    rmsd = np.load(rmsd_fn)
    good_inds = np.where(rmsd<rmsd_cutoff)
    enc = enc[good_inds]
    labels = labels[good_inds]

    inds = np.argsort(labels.flatten())

    act_traj = utils.recon_traj(enc[inds[-n_extreme:]], net, ref_s.top, cm)
    out_fn = os.path.join(outdir, "active.xtc")
    act_traj.save(out_fn)
    for i in range(10):
        out_fn = os.path.join(outdir, "act%d.pdb" % i)
        act_traj[i].save(out_fn)
    act_traj = act_traj.atom_slice(ca_inds)
    act_rmsf = 10*get_rmsf(act_traj)
    out_fn = os.path.join(outdir, "act_rmsf.npy")
    np.save(out_fn, act_rmsf)

    inact_traj = utils.recon_traj(enc[inds[:n_extreme]], net, ref_s.top, cm)
    out_fn = os.path.join(outdir, "inactive.xtc")
    inact_traj.save(out_fn)
    for i in range(10):
        out_fn = os.path.join(outdir, "inact%d.pdb" % i)
        inact_traj[i].save(out_fn)
    inact_traj = inact_traj.atom_slice(ca_inds)
    inact_rmsf = 10*get_rmsf(inact_traj)
    out_fn = os.path.join(outdir, "inact_rmsf.npy")
    np.save(out_fn, inact_rmsf)

    #all_h, x = common_hist([act_rmsf, inact_rmsf], ['act', 'inact'], 20)
    fig = figure(figsize=(4, 8))
    title
    #plot(x, all_h['act'], label='act')
    #plot(x, all_h['inact'], label='inact')
    res_nums = []

    for r in act_traj.top.residues:
        res_nums.append(r.resSeq)

    ax = fig.add_subplot(211)
    plot(res_nums, act_rmsf, label='act')
    plot(res_nums, inact_rmsf, label='inact')
    legend()

    ax = fig.add_subplot(212)
    d = act_rmsf-inact_rmsf
    plot(res_nums, d, 'k')
    out_fn = os.path.join(outdir, "act_minus_inact.npy")
    np.save(out_fn, d)
    show()

    out_fn = os.path.join(outdir, "act_minus_inact.pdb")
    ref_s = ref_s.atom_slice(ca_inds)
    ref_s.save_pdb(out_fn, bfactors=d)
    print("rmsf delta extrema", d.min(), d.mean(), d.max())

def enc_corr(enc):
    n_latent = enc.shape[1]
    corr = []
    for i in range(n_latent):
        for j in range(i+1, n_latent):
            c = pearsonr(enc[:,i], enc[:,j])[0]
            corr.append(c)
    return np.array(corr)


def project_act(lab_v, vars, my_title):
    n_vars = len(vars)
    print(my_title)
    fig = figure(figsize=(4, 4))
    fig.suptitle(my_title)
    for i in range(n_vars):
        v = vars[i]
        n, x = np.histogram(lab_v[v], range=(0, 1), bins=50)
        plot(x[:-1], n, label=v)
        print(v, lab_v[v].mean())
    legend()
    show()


def check_loss(nn_dir):
    i = 2
    fn = os.path.join(nn_dir, "test_loss_%d.npy" % i)
    while os.path.exists(fn):
        d = np.load(fn)
        plot(d, label=str(i))
        i += 1
        fn = os.path.join(nn_dir, "test_loss_%d.npy" % i)
    fn = os.path.join(nn_dir, "test_loss_polish.npy")
    d = load(fn)
    plot(d, label='p')
    legend()
    show()

def clust_encod(nn_dir, n_clusters, vars, lag_times,n_traj_per_var):
    msm_dir = os.path.join(nn_dir, "msm_%d" % n_clusters)
    utils.mkdir(msm_dir)

    enc = utils.load_npy_dir(os.path.join(nn_dir, "encodings"), "*npy")
    enc_v = split_vars(enc, vars)
    n_vars = len(vars)
    #n_traj_per_var = 5

    clusters = cluster.hybrid.hybrid(enc, euc_dist, n_clusters=n_clusters, n_iters=1)
    # clusters.assignments and clusters.centers most relevant vars
    cluster_fn = os.path.join(msm_dir, "clusters.pkl")
    pickle.dump(clusters, open(cluster_fn, 'wb'))

    # assuming 5 traj of equal length per variant, divide into traj
    assigns = clusters.assignments.reshape((n_vars*n_traj_per_var, -1))

    height = 4
    width = height*n_vars
    fig = figure(figsize=(width, height))
    fig.suptitle(nn_dir)
    for i in range(n_vars):
        v = vars[i]
        print("Getting impolied timescales for", v)
        v_assians = assigns[i*n_traj_per_var:(i+1)*n_traj_per_var]

        f = lambda c: msm.builders.normalize(c, prior_counts=1.0/n_clusters, calculate_eq_probs=True)
        imp_times = msm.implied_timescales(v_assians, lag_times, f)
        imp_fn = os.path.join(msm_dir, "%s_imp_norm.npy" % v)
        np.save(imp_fn, imp_times)

        ax = fig.add_subplot(1, n_vars, i+1, aspect='auto')
        for i, t in enumerate(lag_times):
            scatter(t*np.ones(imp_times.shape[1]), imp_times[i])
        title(v)
        ax.set_yscale('log')

        markov_lag = 10
        c = msm.assigns_to_counts(v_assians, 1, max_n_states=n_clusters)
        c_fn = os.path.join(msm_dir, "%s_c_raw_lag%s.npz" % (v, markov_lag))
        scipy.sparse.save_npz(c_fn, c)
        C, T, p = msm.builders.normalize(c, prior_counts=1.0/n_clusters, calculate_eq_probs=True)
        p_fn = os.path.join(msm_dir, "%s_p_norm_lag%d.npy" % (v, markov_lag))
        np.save(p_fn, p)
        T_fn = os.path.join(msm_dir, "%s_T_norm_lag%d.npy" % (v, markov_lag))
        np.save(T_fn, T)
        C_fn = os.path.join(msm_dir, "%s_C_norm_lag%d.npy" % (v, markov_lag))
        np.save(C_fn, C)
    out_fn = os.path.join(msm_dir, "imp_times.png")
    savefig(out_fn)
    show()


