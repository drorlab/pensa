import os
import functools
import pickle
import multiprocessing as mp
import glob

import numpy as np
import mdtraj as md
import torch
import click


class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

#Give me ipath to trajectories, indices, path to master.pdb and cm.npy, and output dir
@click.group()
def cli():
    pass

def _extract_data_from_sim(inputs,pdb_path,inds_fn,whitened_dir,outdir):
    i, traj_fn = inputs
    inds = np.load(inds_fn)
    master = md.load(os.path.join(whitened_dir,"master.pdb"))
    pdb = md.load(pdb_path)
    traj = md.load(traj_fn, top=pdb)
    traj = traj.atom_slice(inds)
    traj = traj.superpose(master, parallel=False)
    data = traj.xyz.reshape((len(traj), 3*master.n_atoms))
    cm = np.load(os.path.join(whitened_dir,"cm.npy"))
    data = data - cm
    torch_traj = torch.from_numpy(data).type(torch.FloatTensor)
    torch.save(torch_traj,os.path.join(outdir,"ID-%s.pt" % i))

@cli.command(name='process_new')
@click.argument('traj_dir')
@click.argument('pdb_path')
@click.argument('inds_fn')
@click.argument('whitened_dir')
@click.argument('outdir')
def extract_data_from_sim(traj_dir, pdb_path, inds_fn, whitened_dir, outdir):
    """ This function converts simulations (xtc files) into input data that
        can be directly fed into a DiffNet.

        traj_dir: Path to directory containing trajectories of a single variant

        pdb_path: Path to pdb that corresponds to trajectories in traj_dir

        inds_fn: File to a numpy array of indices that go with the pdb
                 from pdb_path. Needs to pull equivalent atoms to atoms used
                 in training.

        whitened_dir: Data with processed data that was used for training.

        outdir: Path to an output directory. 
    """

    click.echo(type(traj_dir))
    click.echo(traj_dir)
    click.echo(os.path.join(traj_dir,"*.xtc"))
    traj_fns = glob.glob(os.path.join(traj_dir,"*.xtc"))
    inputs = [(i,j) for i,j in enumerate(traj_fns)]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    n_cores = mp.cpu_count()
    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_extract_data_from_sim, 
                          inds_fn=inds_fn,
                          whitened_dir=whitened_dir,
                          outdir=outdir,pdb_path=pdb_path)
    result = pool.map_async(f, inputs)
    result.wait()
    traj_lens = result.get()
    pool.close()

def chunks(arr, chunk_size):
    """Yield successive chunk_size chunks from arr."""
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]

@cli.command(name='predict_new')
@click.argument('data_dir')
@click.argument('nn_path')
@click.argument('outdir')
@click.option('--save_labels', default=True)
@click.option('--save_latent', default=False)
@click.option('--save_recon', default=False)
def predict(data_dir,nn_path,outdir,save_labels=True,
            save_latent=False,save_recon=False):
    """ Uses an already trained DiffNet to predict on a variant outside
        the training. Requires the variant to be preprocessed for input.
        See extract_data_from_sim for preprocessing.

        data_dir: Directory with pytorch float tensors for each example
                  (i.e. frame of a simulation)

        nn_path: Path to directory with DiffNet training output

        outdir: Directory to output label, latent vectors, and/or reconstructed
                trajectories.
    """

    if not save_labels and not save_latent and not save_recon:
        raise ImproperlyConfigured(
            f'at least one of save_labels, save_latent, or save_recon'
             'must be true.')
    net = pickle.load(open("%s/nn_best_polish.pkl" % nn_path, 'rb'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch_trajs = glob.glob(os.path.join(data_dir,"*.pt"))
    traj_num = 0
    if save_labels:
        os.mkdir(os.path.join(outdir,"labels"))
    if save_latent:
        os.mkdir(os.path.join(outdir,"latent"))
    if save_recon:
        os.mkdir(os.path.join(outdir,"recon"))
    for t in torch_trajs:
        ex = torch.load(t)
        encodings = []
        labels = []
        recon = []
        for batch in chunks(ex,100):
            if use_cuda:
                local_batch = batch.type(torch.cuda.FloatTensor)
            else:
                local_batch = batch.type(torch.FloatTensor)

            local_batch = local_batch.to(device)
            x_pred, latent, class_pred = net(local_batch)

            if save_labels:
                labels.append(class_pred) 
            if save_latent:
                encodings.append(latent)
            if save_recon:
                recon.append(x_pred)

        if save_labels:
            labels = np.concatenate([l.detach().numpy() for l in labels])
            label_dir = os.path.join(outdir, "labels")
            np.save(os.path.join(label_dir,str(traj_num).zfill(6) + ".npy"),
                    labels)

        if save_latent:
            encodings = np.vstack(encodings)
            encodings_dir = os.path.join(outdir, "latent")
            np.save(os.path.join(encodings_dir,str(traj_num).zfill(6) + ".npy"),
                    encodings)   

        if save_recon:
            recon = np.vstack(recon)
            recon_dir = os.path.join(outdir, "recon")
            np.save(os.path.join(recon_dir,str(traj_num).zfill(6) + ".npy"),
                   recon)
        traj_num += 1

if __name__=="__main__":
    cli()
