import os
import pickle
import sys
import multiprocessing as mp
import mdtraj as md
import numpy as np
from . import exmax, nnutils, utils, data_processing
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as torch_data

class Dataset(torch_data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_inds, labels, data):
        'Initialization'
        self.labels = labels
        self.train_inds = train_inds
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_inds)

  def __getitem__(self, index):
        'Generates one sample of data'
        #If data needs to be loaded
        ID = self.train_inds[index]
        if type(self.data) is str:
            # Load data and get label
            X = torch.load(self.data + "/ID-%s" % ID + '.pt')
        else: 
            X = torch.from_numpy(self.data[ID]).type(torch.FloatTensor)
        y = self.labels[ID]
        
        return X, y, ID



class Trainer:

    def __init__(self,job):
        """Object to train your DiffNet.
        
        Parameters:
        -----------
        job : dict 
            Dictionary with all training parameters. See training_dict.txt
            for all keys. All keys are required. See train_submit.py for an
            example.
        """
        self.job = job

    def set_training_data(self, job, train_inds, test_inds, labels, data):
        """Construct generators out of the dataset for training, validation,
        and expectation maximization.

        Parameters
        ----------
        job : dict
            See training_dict.tx for all keys.
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        labels : np.ndarray,
            classification labels used for training
        data : np.ndarray, shape=(n_frames,3*n_atoms) OR str to path
            All data
        """

        batch_size = job['batch_size']
        cpu_cores = job['em_n_cores']
        test_batch_size = job['test_batch_size']
        em_batch_size = job['em_batch_size']
        subsample = job['subsample']
        data_dir = job["data_dir"]

        n_train_inds = len(train_inds)
        random_inds = np.random.choice(np.arange(n_train_inds),int(n_train_inds/subsample),replace=False)
        sampler=torch_data.SubsetRandomSampler(random_inds)

        params_t = {'batch_size': batch_size,
                  'shuffle':False,
                  'num_workers': cpu_cores,
                  'sampler': sampler}

        params_v = {'batch_size': test_batch_size,
                  'shuffle':True,
                  'num_workers': cpu_cores}

        params_e = {'batch_size': em_batch_size,
                  'shuffle':True,
                  'num_workers': cpu_cores}

        n_snapshots = len(train_inds) + len(test_inds)

        training_set = Dataset(train_inds, labels, data)
        training_generator = torch_data.DataLoader(training_set, **params_t)

        validation_set = Dataset(test_inds, labels, data)
        validation_generator = torch_data.DataLoader(validation_set, **params_v)

        em_set = Dataset(train_inds, labels, data)
        em_generator = torch_data.DataLoader(em_set, **params_e)

        return training_generator, validation_generator, em_generator
    
    def em_parallel(self, net, em_generator, train_inds, em_batch_size,
                    indicators, em_bounds, em_n_cores, label_str, epoch):
        """Use expectation maximization to update all training classification
           labels.

        Parameters
        ----------
        net : nnutils neural network object
            Neural network 
        em_generator : Dataset object
            Training data
        train_inds : np.ndarray
            Indices in data that are to be trained on
        em_batch_size : int
            Number of examples that are have their classification labels
             updated in a single round of expectation maximization.
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.
        em_bounds : np.ndarray, shape=(n_variants,2)
            A range that sets what fraction of conformations you
            expect a variant to have biochemical property. Rank order
            of variants is more important than the ranges themselves.
        em_n_cores : int
            CPU cores to use for expectation maximization calculation

        Returns
        -------
        new_labels : np.ndarray, shape=(len(data),)
            Updated classification labels for all training examples
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        n_em = np.ceil(train_inds.shape[0]*1.0/em_batch_size)
        freq_output = np.floor(n_em/10.0)
        train_inds = []
        inputs = []
        i = 0
        ##To save DiffNet labels before each EM update
        pred_labels = -1 * np.ones(indicators.shape[0])
        for local_batch, local_labels, t_inds in em_generator:
            t_inds = np.array(t_inds)
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if hasattr(net, "decode"):
                if hasattr(net, "reparameterize"):
                    x_pred, latent, logvar, class_pred = net(local_batch)
                else:
                    x_pred, latent, class_pred = net(local_batch)
            else:
                class_pred = net(local_batch)
            cur_labels = class_pred.cpu().detach().numpy()
            pred_labels[t_inds] = cur_labels.flatten()
            inputs.append([cur_labels, indicators[t_inds], em_bounds])
            if i % freq_output == 0:
                print("      %d/%d" % (i, n_em))
            i += 1
            train_inds.append(t_inds)

        pred_label_fn = os.path.join(self.job['outdir'],"tmp_labels_%s_%s.npy" % (label_str,epoch))
        np.save(pred_label_fn,pred_labels)
        pool = mp.Pool(processes=em_n_cores)
        res = pool.map(self.apply_exmax, inputs)
        pool.close()

        train_inds = np.concatenate(np.array(train_inds))
        new_labels = -1 * np.ones((indicators.shape[0], 1))
        new_labels[train_inds] = np.concatenate(res)
        return new_labels

    def apply_exmax(self, inputs):
        """Apply expectation maximization to a batch of data.

        Parameters
        ----------
        inputs : list
            list where the 0th index is a list of current classification
            labels of length == batch_size. 1st index is a corresponding
            list of variant simulation indicators. 2nd index is em_bounds.
            
        Returns
        -------
        Updated labels -- length == batch size
        """
        cur_labels, indicators, em_bounds = inputs
        n_vars = em_bounds.shape[0]

        for i in range(n_vars):
            inds = np.where(indicators == i)[0]
            lower = np.int(np.floor(em_bounds[i, 0] * inds.shape[0]))
            upper = np.int(np.ceil(em_bounds[i, 1] * inds.shape[0]))
            cur_labels[inds] = exmax.expectation_range_CUBIC(cur_labels[inds], lower, upper).reshape(cur_labels[inds].shape)

        bad_inds = np.where(np.isnan(cur_labels))
        cur_labels[bad_inds] = 0
        try:
            assert((cur_labels >= 0.).all() and (cur_labels <= 1.).all())
        except AssertionError:
            neg_inds = np.where(cur_labels<0)[0]
            pos_inds = np.where(cur_labels>1)[0]
            bad_inds = neg_inds.tolist() + pos_inds.tolist()
            for iis in bad_inds:
                print("      ", indicators[iis], cur_labels[iis])
            print("      #bad neg, pos", len(neg_inds), len(pos_inds))
            #np.save("tmp.npy", tmp_labels)
            cur_labels[neg_inds] = 0.0
            cur_labels[pos_inds] = 1.0
            #sys.exit(1)
        return cur_labels.reshape((cur_labels.shape[0], 1))

    def train(self, data, training_generator, validation_generator, em_generator,
              targets, indicators, train_inds, test_inds,net, label_str,
              job, lr_fact=1.0):
        """Core method for training

        Parameters
        ----------
        data : np.ndarray, shape=(n_frames,3*n_atoms) OR str to path
            Training data
        training_generator: Dataset object
            Generator to sample training data
        validation_generator: Dataset object
            Generator to sample validation data
        em_generator: Dataset object
            Generator to sample training data in batches for expectation
            maximization
        targets : np.ndarray, shape=(len(data),)
            classification labels used for training
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        net : nnutils neural network object
            Neural network
        label_str: int
            For file naming. Indicates what iteration of training we're
            on. Training goes through several iterations where neural net
            architecture is progressively built deeper.
        job : dict
            See training_dict.tx for all keys.
        lr_fact : float
            Factor to multiply the learning rate by.

        Returns
        -------
        best_nn : nnutils neural network object
            Neural network that has the lowest reconstruction error
            on the validation set.
        targets : np.ndarry, shape=(len(data),)
            Classification labels after training.
        """
        job = self.job
        do_em = job['do_em']
        n_epochs = job['n_epochs']
        lr = job['lr'] * lr_fact
        subsample = job['subsample']
        batch_size = job['batch_size']
        batch_output_freq = job['batch_output_freq']
        epoch_output_freq = job['epoch_output_freq']
        test_batch_size = job['test_batch_size']
        em_bounds = job['em_bounds']
        nntype = job['nntype']
        em_batch_size = job['em_batch_size']
        em_n_cores = job['em_n_cores']
        outdir = job['outdir'] 

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        n_test = test_inds.shape[0]
        lam_cls = 1.0
        lam_corr = 1.0

        n_batch = np.ceil(train_inds.shape[0]*1.0/subsample/batch_size)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        bce = nn.BCELoss()
        training_loss_full = []
        test_loss_full = []
        epoch_test_loss = []
        best_loss = np.inf
        best_nn = None 
        for epoch in range(n_epochs):
            # go through mini batches
            running_loss = 0
            i = 0
            for local_batch, local_labels, _ in training_generator:
                if use_cuda:
                    local_labels = local_labels.type(torch.cuda.FloatTensor)
                else:
                    local_labels = local_labels.type(torch.FloatTensor)
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                optimizer.zero_grad()    
                x_pred, latent, class_pred = net(local_batch)
                loss = nnutils.my_mse(local_batch, x_pred)
                loss += nnutils.my_l1(local_batch, x_pred)
                if class_pred is not None:
                    loss += bce(class_pred, local_labels).mul_(lam_cls)
                #Minimize correlation between latent variables
                n_feat = net.sizes[-1]
                my_c00 = torch.einsum('bi,bo->io', (latent, latent)).mul(1.0/local_batch.shape[0])
                my_mean = torch.mean(latent, 0)
                my_mean = torch.einsum('i,o->io', (my_mean, my_mean))
                ide = np.identity(n_feat)
                if use_cuda:
                    ide = torch.from_numpy(ide).type(torch.cuda.FloatTensor)
                else:
                    ide = torch.from_numpy(ide).type(torch.FloatTensor)
                #ide = Variable(ide)
                #ide = torch.from_numpy(np.identity(n_feat))
                #ide = ide.to(device)
                zero_inds = np.where(1-ide.cpu().numpy()>0)
                corr_penalty = nnutils.my_mse(ide[zero_inds], my_c00[zero_inds]-my_mean[zero_inds])
                loss += corr_penalty
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i%batch_output_freq == 0:
                    train_loss = running_loss
                    if i != 0:
                        train_loss /= batch_output_freq
                    training_loss_full.append(train_loss)

                    test_loss = 0
                    for local_batch, local_labels, _ in validation_generator:
                        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                        x_pred, latent, class_pred = net(local_batch)
                        loss = nnutils.my_mse(local_batch,x_pred)
                        test_loss += loss.item() * local_batch.shape[0] # mult for averaging across samples, as in train_loss
                    #print("        ", test_loss)
                    test_loss /= n_test # division averages across samples, as in train_loss
                    test_loss_full.append(test_loss)
                    print("    [%s %d, %5d/%d] train loss: %0.6f    test loss: %0.6f" % (label_str, epoch, i, n_batch, train_loss, test_loss))
                    running_loss = 0

                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_nn = copy.deepcopy(net)
                i += 1

            if do_em and hasattr(nntype, "classify"):
                print("    Doing EM")
                targets = self.em_parallel(net, em_generator, train_inds,
                                    em_batch_size, indicators, em_bounds,
                                    em_n_cores, label_str, epoch)
                training_generator, validation_generator, em_generator = \
                    self.set_training_data(job, train_inds, test_inds, targets, data)

            if epoch % epoch_output_freq == 0:
                print("my_l1", nnutils.my_l1(local_batch, x_pred))
                print("corr penalty",corr_penalty)
                print("classify", bce(class_pred, local_labels).mul_(lam_cls))
                print("my_mse", nnutils.my_mse(local_batch, x_pred))
                epoch_test_loss.append(test_loss)
                out_fn = os.path.join(outdir, "epoch_test_loss_%s.npy" % label_str)
                np.save(out_fn, epoch_test_loss)
                out_fn = os.path.join(outdir, "training_loss_%s.npy" % label_str)
                np.save(out_fn, training_loss_full)
                out_fn = os.path.join(outdir, "test_loss_%s.npy" % label_str)
                np.save(out_fn, test_loss_full)
            # nets need be on cpu to load multiple in parallel, e.g. with multiprocessing
                net.cpu()
                out_fn = os.path.join(outdir, "nn_%s_e%d.pkl" % (label_str, epoch))
                pickle.dump(net, open(out_fn, 'wb'))
                if use_cuda:
                    net.cuda()
                if hasattr(nntype, "classify"):
                    out_fn = os.path.join(outdir, "tmp_targets_%s_%s.npy" % (label_str,epoch))
                    np.save(out_fn, targets)

            # save best net every epoch
            best_nn.cpu()
            out_fn = os.path.join(outdir, "nn_best_%s.pkl" % label_str)
            pickle.dump(best_nn, open(out_fn, 'wb'))
            if use_cuda:
                best_nn.cuda()
        return best_nn, targets    

    def get_targets(self,act_map,indicators,label_spread=None):
        """Convert variant indicators into classification labels.

        Parameters
        ----------
        act_map : np.ndarray, shape=(n_variants,)
            Initial classification labels to give each variant.
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.

        Returns
        -------
        targets : np.ndarry, shape=(len(data),)
            Classification labels for training.
        """
        targets = np.zeros((len(indicators), 1))
        print(targets.shape)
        if label_spread == 'gaussian':
            targets = np.array([np.random.normal(act_map[i],0.1) for i in indicators])
            zero_inds = np.where(targets < 0)[0]
            targets[zero_inds] = 0
            one_inds = np.where(targets > 1)[0]
            targets[one_inds] = 1
        elif label_spread == 'uniform':
            targets = np.vstack([np.random.uniform() for i in targets])
        elif label_spread == 'bimodal':
            targets = np.array([np.random.normal(0.8, 0.1) if np.random.uniform() < act_map[i]
                                else np.random.normal(0.2, 0.1) for i in indicators])
            zero_inds = np.where(targets < 0)[0]
            targets[zero_inds] = 0
            one_inds = np.where(targets > 1)[0]
            targets[one_inds] = 1
        else:
            targets[:, 0] = act_map[indicators]    
        return targets

    def split_test_train(self,n,frac_test):
        """Split data into training and validation sets.

        Parameters
        ----------
        n : int
            number of data points
        frac_test : float between 0 and 1
            Fraction of dataset to reserve for validation set

        Returns
        -------
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        """
        n_test = int(n*frac_test)
       
        inds = np.arange(n)
        np.random.shuffle(inds)
        train_inds = inds[:-n_test]
        test_inds = inds[-n_test:]

        return train_inds, test_inds
    
    def run(self, data_in_mem=False):
        """Wrapper for running the training code

        Parameters
        ----------
        data_in_mem: boolean
            If true, load all training data into memory. Training faster this way.
        
        Returns
        -------
        net : nnutils neural network object
            Trained DiffNet
        """
        job = self.job 
        data_dir = job['data_dir']
        outdir = job['outdir']
        n_latent = job['n_latent']
        layer_sizes = job['layer_sizes']
        nntype = job['nntype']
        frac_test = job['frac_test']
        act_map = job['act_map']

        use_cuda = torch.cuda.is_available()
        print("Using cuda? %s" % use_cuda)

        indicator_dir = os.path.join(data_dir, "indicators")
        indicators = utils.load_npy_dir(indicator_dir, "*.npy")
        indicators = np.array(indicators, dtype=int)
        
        if 'label_spreading' in job.keys():
            targets = self.get_targets(act_map,indicators,
                                       label_spread=job['label_spreading'])
        else:
            targets = self.get_targets(act_map,indicators)
        n_snapshots = len(indicators)
        np.save(os.path.join(outdir, 'initial_targets.npy'), targets)

        train_inds, test_inds = self.split_test_train(n_snapshots,frac_test)
        if data_in_mem:
            xtc_dir = os.path.join(data_dir,"aligned_xtcs")
            top_fn = os.path.join(data_dir, "master.pdb")
            master = md.load(top_fn)
            data = utils.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
        else:
            data = os.path.join(data_dir, "data")

        training_generator, validation_generator, em_generator = \
            self.set_training_data(job, train_inds, test_inds, targets, data)

        print("  data generators created")

        print("# of examples", targets.shape)

        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        uwm = np.load(uwm_fn)
        cm = np.load(cm_fn).flatten()

        n_train = train_inds.shape[0]
        n_test = test_inds.shape[0]
        out_fn = os.path.join(outdir, "train_inds.npy")
        np.save(out_fn, train_inds)
        out_fn = os.path.join(outdir, "test_inds.npy")
        np.save(out_fn, test_inds)
        print("    n train/test", n_train, n_test)

        if hasattr(nntype, 'split_inds'):
            inds1 = job['inds1']
            inds2 = job['inds2']
            old_net = nntype(layer_sizes[0:2],inds1,inds2,wm,uwm)
        else:
            old_net = nntype(layer_sizes[0:2],wm,uwm)
        old_net.freeze_weights()

        for cur_layer in range(2,len(layer_sizes)):
            if hasattr(nntype, 'split_inds'):
                net = nntype(layer_sizes[0:cur_layer+1],inds1,inds2,wm,uwm)
            else:
                net = nntype(layer_sizes[0:cur_layer+1],wm,uwm)
            net.freeze_weights(old_net)
            if use_cuda:
                net.cuda()
            net, targets = self.train(data, training_generator, 
                               validation_generator, em_generator,
                               targets, indicators, train_inds,
                               test_inds, net, str(cur_layer), job)
            #Might make sense to make this optional
            training_generator, validation_generator, em_generator = \
                self.set_training_data(job, train_inds, test_inds, targets, data)          
            old_net = net

        #Polishing
        net.unfreeze_weights()
        if use_cuda:
            net.cuda()
        net, targets = self.train(data, training_generator, validation_generator,
                               em_generator, targets, indicators, train_inds,
                               test_inds, net, "polish", job, lr_fact=0.1)
        return net
