# supervised autoencoders
import mdtraj as md
import multiprocessing as mp
import numpy as np
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class split_ae(nn.Module):
    """Unsupervised autoencoder with a split input (i.e. 2 encoders)

       Parameters
       ----------
       layer_sizes : list
           List of integers indicating the size of each layer in the 
           encoder including the latent layer. First two must be identical.
       inds1 : np.ndarray 
             Indices in the training input array that go into encoder1.
       inds2 : np.ndarray
             Indices in the training input array that go into encoder2.
       wm : np.ndarray, shape=(n_inputs,n_inputs)
            Whitening matrix -- is applied to input data
       uwm : np.ndarray, shape=(n_inputs,n_inputs)
            unwhitening matrix
       """

    def __init__(self,layer_sizes,inds1,inds2,wm,uwm):
        super(split_ae, self).__init__()
        self.sizes = layer_sizes
        self.n = len(self.sizes)
        self.inds1 = inds1
        self.inds2 = inds2
        self.n_features = len(self.inds1)+len(self.inds2)
        self.wm1 = wm[self.inds1[:,None],self.inds1]
        self.wm2 = wm[self.inds2[:,None],self.inds2]
        self.uwm = uwm
        self.ratio = len(inds1)/(len(inds1)+len(inds2))

        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()
        self.encoder1.append(nn.Linear(len(inds1),len(inds1)))
        self.encoder2.append(nn.Linear(len(inds2),len(inds2)))
        for i in range(1,self.n-1):
            small_layer_in = int(np.ceil(self.sizes[i]*self.ratio))
            print(small_layer_in)
            small_layer_out = int(np.ceil(self.sizes[i+1]*self.ratio))
            print(small_layer_out)
            big_layer_in = int(np.floor(self.sizes[i] * (1-self.ratio)))
            print(big_layer_in)
            big_layer_out = int(np.floor(self.sizes[i+1] * (1-self.ratio)))
            print(big_layer_out)
            #if small_layer_in < 3:
            #    small_layer_in = 3
            #    big_layer_in = self.sizes[i]-3
            #if small_layer_out < 3:
            #    small_layer_out = 3
            #    big_layer_out = self.sizes[i]-3

            self.encoder1.append(nn.Linear(small_layer_in, small_layer_out))
            self.encoder2.append(nn.Linear(big_layer_in,big_layer_out))

        self.decoder = nn.ModuleList()
        for i in range(self.n-1,0,-1):
            self.decoder.append(nn.Linear(self.sizes[i], self.sizes[i-1]))

    @property
    def split_inds(self):
        return True

    def freeze_weights(self,old_net=None):
        """Procedure to make the whitening matrix and unwhitening matrix
           as untrainable layers. Additionally, freezes weights associated
           with a previously learned encoder layer.

        Parameters
        ----------
        old_net : split_ae object
            Previously trained network with overlapping architecture. Weights
            learned in this previous networks encoder will be frozen in the
            new network.
        """
        vwm = Variable(torch.from_numpy(self.wm1).type(torch.FloatTensor))
        self.encoder1[0].weight.data = vwm
        vz = Variable(torch.from_numpy(np.zeros(len(self.inds1))).type(torch.FloatTensor))
        self.encoder1[0].bias.data = vz
        vwm2 = Variable(torch.from_numpy(self.wm2).type(torch.FloatTensor))
        self.encoder2[0].weight.data = vwm2
        vz2 = Variable(torch.from_numpy(np.zeros(len(self.inds2))).type(torch.FloatTensor))
        self.encoder2[0].bias.data = vz2
        for p in self.encoder1[0].parameters():
            p.requires_grad = False
        for p in self.encoder2[0].parameters():
            p.requires_grad = False
        self.decoder[-1].weight.data = Variable(torch.from_numpy(self.uwm).type(torch.FloatTensor))
        self.decoder[-1].bias.data = Variable(torch.from_numpy(np.zeros(self.n_features)).type(torch.FloatTensor))
        for p in self.decoder[-1].parameters():
            p.requires_grad = False
        
        if old_net:
            n_old = len(old_net.encoder1)
            for i in range(1,n_old):
                self.encoder1[i].weight.data = old_net.encoder1[i].weight.data
                self.encoder1[i].bias.data = old_net.encoder1[i].bias.data
                for p in self.encoder1[i].parameters():
                    p.requires_grad = False
                self.encoder2[i].weight.data = old_net.encoder2[i].weight.data
                self.encoder2[i].bias.data = old_net.encoder2[i].bias.data
                for p in self.encoder2[i].parameters():
                    p.requires_grad = False

    def unfreeze_weights(self):
        """Makes all encoders weights trainable.
        """
        n_old = len(self.encoder1)
        for i in range(1,n_old):
           for p in self.encoder1[i].parameters():
               p.requires_grad = True
           for p in self.encoder2[i].parameters():
               p.requires_grad = True

    def encode(self,x):
        """Pass the data through the encoder to the latent layer.

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Input data for a given sample

        Returns
        -------
        lat1 : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector associated with encoder1
        lat2 : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector associated with encoder2
        """
        x1 = x[:,self.inds1]
        x2 = x[:,self.inds2]
        lat1 = self.encoder1[0](x1)
        lat2 = self.encoder2[0](x2)
        for i in range(1, self.n-1):
            lat1 = F.leaky_relu(self.encoder1[i](lat1))
            lat2 = F.leaky_relu(self.encoder2[i](lat2))
        return lat1, lat2

    def decode(self,latent):
        """Pass the latent space vector through the decoder

        Parameters
        ----------
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        """
        recon = latent
        for i in range(self.n-2):
            recon = F.leaky_relu(self.decoder[i](recon))
        recon = self.decoder[-1](recon)
        return recon

    def forward(self,x):
        """Pass data through the entire network
        
        Parameters 
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Input data for a given sample

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector
        """
        lat1, lat2 = self.encode(x)
        latent = torch.cat((lat1,lat2),1)
        recon = self.decode(latent)

        return recon, latent, None

class split_sae(split_ae):
    """Supervised autoencoder with split architecture"""
    def __init__(self, layer_sizes,inds1,inds2,wm,uwm):
        super(split_sae, self).__init__(layer_sizes,inds1,inds2,wm,uwm)

        self.classifier = nn.Linear(self.encoder1[-1].weight.data.shape[0], 1)

    def classify(self, latent):
        """Perfom classification task using latent space representation
        
        Parameters
        ----------
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        Value between 0 and 1
        """
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        """Pass data through the entire network

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Input data for a given sample

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector
        label : 
            Value between 0 and 1
        """
        lat1, lat2 = self.encode(x)
        
        label = self.classify(lat1)

        latent = torch.cat((lat1,lat2),1)

        recon = self.decode(latent)
        return recon, latent, label


class ae(nn.Module):
    """Unsupervised autoencoder

       Parameters
       ----------
       layer_sizes : list
           List of integers indicating the size of each layer in the
           encoder including the latent layer. First two must be identical.
       wm : np.ndarray, shape=(n_inputs,n_inputs)
            Whitening matrix -- is applied to input data
       uwm : np.ndarray, shape=(n_inputs,n_inputs)
            unwhitening matrix
    """
    def __init__(self, layer_sizes,wm,uwm):
        super(ae, self).__init__()
        self.sizes = layer_sizes
        self.n = len(self.sizes)
        self.wm = wm
        self.uwm = uwm

        self.encoder = nn.ModuleList()
        for i in range(self.n-1):
            self.encoder.append(nn.Linear(self.sizes[i], self.sizes[i+1]))

        self.decoder = nn.ModuleList()
        for i in range(self.n-1, 0, -1):
            self.decoder.append(nn.Linear(self.sizes[i], self.sizes[i-1]))

    def freeze_weights(self,old_net=None):
        """Procedure to make the whitening matrix and unwhitening matrix
           as untrainable layers. Additionally, freezes weights associated
           with a previously learned encoder layer.

        Parameters
        ----------
        old_net : ae object
            Previously trained network with overlapping architecture. Weights
            learned in this previous networks encoder will be frozen in the
            new network.
        """
        self.encoder[0].weight.data = Variable(torch.from_numpy(self.wm).type(torch.FloatTensor))
        self.encoder[0].bias.data = Variable(torch.from_numpy(np.zeros(len(self.wm))).type(torch.FloatTensor))
        for p in self.encoder[0].parameters():
            p.requires_grad = False
        self.decoder[-1].weight.data = Variable(torch.from_numpy(self.uwm).type(torch.FloatTensor))
        self.decoder[-1].bias.data = Variable(torch.from_numpy(np.zeros(len(self.uwm))).type(torch.FloatTensor))
        for p in self.decoder[-1].parameters():
            p.requires_grad = False

        if old_net:
            n_old = len(old_net.encoder)
            for i in range(1,n_old):
                self.encoder[i].weight.data = old_net.encoder[i].weight.data
                self.encoder[i].bias.data = old_net.encoder[i].bias.data
                for p in self.encoder[i].parameters():
                    p.requires_grad = False

    def unfreeze_weights(self):
        """Makes all encoders weights trainable.
        """
        n_old = len(self.encoder)
        for i in range(1,n_old):
           for p in self.encoder[i].parameters():
               p.requires_grad = True

    def encode(self, x):
        """Pass the data through the encoder to the latent layer.

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Input data for a given sample

        Returns
        -------
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector associated with encoder1
        """
        # whiten, without applying non-linearity
        latent = self.encoder[0](x)

        # do non-linear layers
        for i in range(1, self.n-1):
            latent = F.leaky_relu(self.encoder[i](latent))

        return latent

    def decode(self, x):
        """Pass the latent space vector through the decoder

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        """
        # do non-linear layers
        recon = x
        for i in range(self.n-2):
            recon = F.leaky_relu(self.decoder[i](recon))
        
        # unwhiten, without applying non-linearity
        recon = self.decoder[-1](recon)

        return recon

    def forward(self, x):
        """Pass data through the entire network

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Input data for a given sample

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        # None for labels, so number returns same as sae class
        return recon, latent, None


# build classifier based on latent representation from an AE
class classify_ae(nn.Module):
    """Logistic Regression model
       
       Parameters
       ----------
       n_latent : int
           Number of latent variables
    """
    def __init__(self, n_latent):
        super(classify_ae, self).__init__()
        self.n_latent = n_latent

        self.fc1 = nn.Linear(self.n_latent, 1)

    def classify(self, x):
        """Perfom classification task using latent space representation

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        Value between 0 and 1
        """
        return torch.sigmoid(self.fc1(x))

    def forward(self, x):
        """Perfom classification task using latent space representation

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        Value between 0 and 1
        """
        return self.classify(x)


class sae(ae):
    """Supervised autoencoder

    Parameters
    ----------
    layer_sizes : list
           List of integers indicating the size of each layer in the
           encoder including the latent layer. First two must be identical.
       wm : np.ndarray, shape=(n_inputs,n_inputs)
            Whitening matrix -- is applied to input data
       uwm : np.ndarray, shape=(n_inputs,n_inputs)
            unwhitening matrix
    """
    def __init__(self, layer_sizes, wm, uwm):
        super(sae, self).__init__(layer_sizes,wm,uwm)
        
        self.classifier = nn.Linear(self.sizes[-1], 1)

    def classify(self, latent):
        """Perfom classification task using latent space representation

        Parameters
        ----------
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        Value between 0 and 1
        """
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        """Pass through the entire network

        Parameters
        ----------
        x : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector

        Returns
        -------
        recon : torch.cuda.FloatTensor or torch.FloatTensor
            Reconstruction of the original input data
        latent : torch.cuda.FloatTensor or torch.FloatTensor
            Latent space vector
        label :
            Value between 0 and 1
        """
        latent = self.encode(x)
        label = self.classify(latent)
        recon = self.decode(latent)
        return recon, latent, label


class vae(ae):
    def __init__(self, layer_sizes):
        super(vae, self).__init__(layer_sizes)

        # last layer of encoder is mu, also need logvar of equal size
        self.logvar = nn.Linear(self.sizes[-2], self.sizes[-1])

    def encode(self, x):
        # whiten, without applying non-linearity
        latent = self.encoder[0](x)

        # do non-linear layers
        for i in range(1, self.n-2):
            latent = F.leaky_relu(self.encoder[i](latent))

        mu = F.leaky_relu(self.encoder[-1](latent))
        logvar = F.leaky_relu(self.logvar(latent))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # None for labels, so number returns same as sae class
        return recon, mu, logvar, None


class svae(vae):
    def __init__(self, layer_sizes):
        super(svae, self).__init__(layer_sizes)
        
        self.classifier = nn.Linear(self.sizes[-1], 1)

    def classify(self, latent):
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        label = self.classify(z)
        recon = self.decode(z)
        # None for labels, so number returns same as sae class
        return recon, mu, logvar, label


def my_mse(x, x_recon):
    """Calculate mean squared error loss

    Parameters
    ----------
    x : torch.cuda.FloatTensor or torch.FloatTensor
        Input data
    x_recon : torch.cuda.FloatTensor or torch.FloatTensor
        Reconstructed input

    Returns
    -------
    torch.cuda.FloatTensor or torch.FloatTensor
    """
    return torch.mean(torch.pow(x-x_recon, 2))


def my_l1(x, x_recon):
    """Calculate l1 loss

    Parameters
    ----------
    x : torch.cuda.FloatTensor or torch.FloatTensor
        Input data
    x_recon : torch.cuda.FloatTensor or torch.FloatTensor
        Reconstructed input

    Returns
    -------
    torch.cuda.FloatTensor or torch.FloatTensor
    """
    return torch.mean(torch.abs(x-x_recon))

def chunks(arr, chunk_size):
    """Yield successive chunk_size chunks from arr."""
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]

def split_inds(pdb,resnum,focus_dist):
        """Identify indices close and far from a residue of interest.
           Each index corresponds to an X,Y, or Z coordinate of an atom
           in the pdb.

        Parameters
        ----------
        pdb : md.Trajectory object
            Structure used to find close/far indices.
        resnum : integer
            The residue number of interest.
        focus_dist : float (nannmeters)
            All indices within this distance of resnum will be selected
            as close indices.

        Returns
        -------
        close_xyz_inds : np.ndarray
            Indices of x,y,z positions of atoms in pdb that are close
            to resnum.
        non_close_xyz_inds : np.ndarray
            Indices of x,y,z positions of atoms in pdb that are not
            close to resnum.
        """
        res_atoms = pdb.topology.select("resSeq %s" % resnum)
        dist_combos = [res_atoms,np.arange(pdb.n_atoms)]
        dist_combos = np.array(list(itertools.product(*dist_combos)))

        dpdb = md.compute_distances(pdb,dist_combos)
        ind_loc = np.where(dpdb.flatten()<focus_dist)[0]
        inds = np.unique(dist_combos[ind_loc].flatten())

        close_xyz_inds = []
        for i in inds:
            close_xyz_inds.append(i*3)
            close_xyz_inds.append((i*3)+1)
            close_xyz_inds.append((i*3)+2)
        all_inds = np.arange((pdb.n_atoms*3))
        non_close_xyz_inds = np.setdiff1d(all_inds,close_xyz_inds)

        return np.array(close_xyz_inds), non_close_xyz_inds

