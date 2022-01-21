import numpy as np
from tqdm import tqdm
from pensa.features import *
from pensa.statesinfo import *
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import pyemma
from pyemma.util.contexts import settings
import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# -- Functions to calculate SSI statistics across paired ensembles --


def unc_relative_entropy_analysis(features_a, features_b, all_data_a, all_data_b, bin_width=None, bin_num=10, block_length=None, verbose=True, override_name_check=False):
    """
    Relative entropy metrics for truncated data on each feature from two ensembles.

    Parameters
    ----------
    features_a : list of str
        Feature names of the first ensemble. 
        Can be obtained from features object via .describe().
    features_b : list of str
        Feature names of the first ensemble. 
        Can be obtained from features object via .describe().
        Must be the same as features_a. Provided as a sanity check. 
    all_data_a : float array
        Trajectory data from the first ensemble. Format: [frames,frame_data].
    all_data_b : float array
        Trajectory data from the second ensemble. Format: [frames,frame_data].
    bin_width : float, default=None
        Bin width for the axis to compare the distributions on. 
        If bin_width is None, bin_num (see below) bins are used and the width is determined from the common histogram.
    bin_num : int, default=10
        Number of bins for the axis to compare the distributions on (only if bin_width=None).
    block_length : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : bool, default=True
        Print intermediate results.
    override_name_check : bool, default=False
        Only check number of features, not their names.

    Returns
    -------
    data_names : list of str
        Feature names.
    data_jsdist : float array
        Jensen-Shannon distance for each feature.
    data_kld_ab : float array
        Kullback-Leibler divergences of data_a wrt to data_b.
    data_kld_ba : float array
        Kullback-Leibler divergences of data_b wrt to data_a.

    """

    all_data_a, all_data_b = all_data_a.T, all_data_b.T
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(features_a) == len(features_b)
    else:
        assert features_a == features_b
    assert all_data_a.shape[0] == all_data_b.shape[0] 
    # Extract the names of the features
    data_names = features_a
    # Initialize relative entropy and average value
    data_jsdist = np.zeros(len(data_names))
    data_kld_ab = np.zeros(len(data_names))
    data_kld_ba = np.zeros(len(data_names))
    data_avg    = np.zeros(len(data_names))
    # Loop over all features
    for i in range(len(all_data_a)):
        if block_length is not None:
            BL1 = block_length[0]
            BL2 = block_length[1]
            data_a = all_data_a[i][BL1:BL2]
            data_b = all_data_b[i][BL1:BL2]
            
        else:
            data_a = all_data_a[i]
            data_b = all_data_b[i]
        # Combine both data sets
        data_both = np.concatenate((data_a,data_b))
        data_avg[i] = np.mean(data_both)
        # Get bin values for all histograms from the combined data set
        if bin_width is None:
            bins = bin_num
        else:
            bins_min = np.min( data_both )
            bins_max = np.max( data_both )
            bins = np.arange(bins_min,bins_max,bin_width)
        # Calculate histograms for combined and single data sets
        histo_both = np.histogram(data_both, bins = bins, density = True)
        histo_a = np.histogram(data_a, density = True, bins = histo_both[1])
        distr_a = histo_a[0] / np.sum(histo_a[0])
        histo_b = np.histogram(data_b, density = True, bins = histo_both[1])
        distr_b = histo_b[0] / np.sum(histo_b[0])
        # Calculate relative entropies between the two data sets (Kullback-Leibler divergence)
        data_kld_ab[i] = np.sum( sp.special.kl_div(distr_a,distr_b) )
        data_kld_ba[i] = np.sum( sp.special.kl_div(distr_b,distr_a) )
        # Calculate the Jensen-Shannon distance
        data_jsdist[i] = scipy.spatial.distance.jensenshannon(distr_a, distr_b, base=2.0)
        # Print information
        if verbose:
            print(i,'/',len(all_data_a),':', data_names[i]," %1.2f"%data_avg[i],
                  " %1.2f %1.2f %1.2f"%(data_jsdist[i],data_kld_ab[i],data_kld_ba[i]))   
    return data_names, data_jsdist, data_kld_ab, data_kld_ba



def unc_ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbc=True, 
                              bin_no=180, block_length=None, verbose=True, write_plots=None, override_name_check=False):
    """
    State Specific Information statistic for truncated data on each feature across two ensembles.


    Parameters
    ----------
    features_a : list of str
        Feature names of the first ensemble. 
    features_b : list of str
        Feature names of the first ensemble. 
        Must be the same as features_a. Provided as a sanity check. 
    all_data_a : float array
        Trajectory data from the first ensemble. Format: [frames,frame_data].
    all_data_b : float array
        Trajectory data from the second ensemble. Format: [frames,frame_data].
    torsions : str
        Torsion angles to use for SSI, including backbone - 'bb', and sidechain - 'sc'. 
        Default is None.
    pocket_occupancy : bool, optional
        Set to 'True' if the data input is pocket occupancy distribution.
        The default is None.
    pbc : bool, optional
        If true, the apply periodic bounary corrections on angular distribution inputs.
        The input for periodic correction must be radians. The default is True.
    bin_no : int, default=10
        Number of bins for the distribution histogram when performing clustering.
        The default is 180.
    block_length : TYPE, optional
        DESCRIPTION. The default is None.
    verbose : bool, default=True
        Print intermediate results.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    override_name_check : bool, default=False
        Only check number of features, not their names.   
        
    Returns
    -------
    data_names : list of str
        Feature names.
    data_ssi : float array
        State Specific Information statistics for each feature.

    """

    
    # Get the multivariate timeseries data
    if torsions is None:
         mv_res_feat_a, mv_res_data_a = features_a,all_data_a
         mv_res_feat_b, mv_res_data_b = features_b,all_data_b
    else:
         mv_res_feat_a, mv_res_data_a = get_multivar_res_timeseries(features_a,all_data_a,torsions+'-torsions',write=False,out_name='')
         mv_res_feat_b, mv_res_data_b = get_multivar_res_timeseries(features_b,all_data_b,torsions+'-torsions',write=False,out_name='')
    
         mv_res_feat_a, mv_res_data_a = mv_res_feat_a[torsions+'-torsions'], mv_res_data_a[torsions+'-torsions']
         mv_res_feat_b, mv_res_data_b = mv_res_feat_b[torsions+'-torsions'], mv_res_data_b[torsions+'-torsions']
    
    # Assert that the features are the same and data sets have same number of features
    if override_name_check:
        assert len(mv_res_feat_a) == len(mv_res_feat_b)
    else:
        assert mv_res_feat_a == mv_res_feat_b
    assert mv_res_data_a.shape[0] == mv_res_data_b.shape[0] 
    # Extract the names of the features
    data_names = mv_res_feat_a
    # Initialize relative entropy and average value
    data_ssi = np.zeros(len(data_names))
    # Loop over all features    
    for residue in range(len(mv_res_data_a)):
        data_a = mv_res_data_a[residue]
        data_b = mv_res_data_b[residue]

        combined_dist=[]
        
        if block_length is not None:
            BL1 = block_length[0]
            BL2 = block_length[1]
            for dist_no in range(len(data_a)):
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                data_both = list(data_a[dist_no][BL1:BL2]) + list(data_b[dist_no][BL1:BL2])      
                combined_dist.append(data_both)
            ## Saving distribution length
            traj1_len = len(data_a[dist_no][BL1:BL2])       
                
        else:
            for dist_no in range(len(data_a)):
                # # # combine the ensembles into one distribution (condition_a + condition_b)
                data_both = list(data_a[dist_no]) + list(data_b[dist_no])      
                combined_dist.append(data_both)
            traj1_len = len(data_a[dist_no]) 


        if pbc is True:
            feat_distr = [correct_angle_periodicity(distr) for distr in combined_dist]
        else:
            feat_distr = combined_dist        
        
        cluster=1
        if pocket_occupancy is True: 
            ## Define states for water occupancy 
            feat_states = [[-0.5,0.5,1.5]]    
        else:     
            feat_states=[]
            for dim_num in range(len(feat_distr)):
                if write_plots is True:
                    plot_name = data_names[residue]
                else:
                    plot_name = None
                try:
                    feat_states.append(determine_state_limits(feat_distr[dim_num], 
                                                              traj1_len, 
                                                              write_plots=write_plots, 
                                                              write_name=plot_name,
                                                              gauss_bins=bin_no))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
                    cluster=0
                    
        if cluster==0:
            SSI = -1
            data_ssi[residue] = SSI
            if verbose is True:
                print(data_names[residue],data_ssi[residue])
        else:
            H_feat=calculate_entropy(feat_states,feat_distr) 
                    
            if H_feat != 0:
                ##calculating the entropy for set_distr_b
                ## if no dist (None) then apply the binary dist for two simulations
                ens_distr=[[0.5]*traj1_len + [1.5]*int(len(feat_distr[0])-traj1_len)]
                ens_states= [[0,1,2]]  
                    
                traj_1_fraction = traj1_len/len(feat_distr[0])
                traj_2_fraction = 1 - traj_1_fraction
                norm_factor = -1*traj_1_fraction*math.log(traj_1_fraction,2) - 1*traj_2_fraction*math.log(traj_2_fraction,2)
                H_ens = norm_factor
                
                featens_joint_states= feat_states + ens_states
                featens_joint_distr= feat_distr + ens_distr
                H_featens=calculate_entropy(featens_joint_states,featens_joint_distr)
        
                SSI = ((H_feat + H_ens) - H_featens)/norm_factor
            
            data_ssi[residue] = SSI
                
            if verbose is True:
                print(data_names[residue],data_ssi[residue])
            
            
    return data_names, data_ssi



def ssi_block_analysis(a_rec_feat, b_rec_feat,
                       a_rec_data, b_rec_data,
                       torsions='sc', blockanlen=10000,
                       cumdist=False, verbose=True):
    
    """
    Block analysis on the State Specific Information statistic for each feature across two ensembles.


    Parameters
    ----------
    a_rec_feat : TYPE
        DESCRIPTION.
    b_rec_feat : TYPE
        DESCRIPTION.
    a_rec_data : TYPE
        DESCRIPTION.
    b_rec_data : TYPE
        DESCRIPTION.
    torsions : TYPE, optional
        DESCRIPTION. The default is 'sc'.
    blockanlen : TYPE, optional
        DESCRIPTION. The default is 10000.
    cumdist : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
        
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    ssi_blocks=[]
    block_lengths=[]
    frameno=0
    while frameno <= len(len(a_rec_data[torsions+'-torsions'])):
        if cumdist is True:
            block_lengths.append([0,frameno+blockanlen])
        else:
            block_lengths.append([frameno+1,frameno+blockanlen])
        frameno+=blockanlen
       

    for bl in block_lengths:
        print('block length = ', bl)
        ssi_names, data_ssi = unc_ssi_ensemble_analysis(a_rec_feat, b_rec_feat,
                                                        a_rec_data, b_rec_data,
                                                        torsions=torsions, 
                                                        block_length=bl, 
                                                        verbose=True)
        
        ssi_blocks.append(data_ssi)

    np.save('ssi_bl'+str(blockanlen),np.transpose(np.array(ssi_blocks)))
    
    return np.transpose(ssi_names), np.transpose(ssi_blocks)



def relen_block_analysis(a_rec_feat, b_rec_feat,
                         a_rec_data, b_rec_data, 
                         blockanlen=10000, cumdist=False, verbose=True):
    """
    Block analysis on the relative entropy metrics for each feature from two ensembles.
    

    Parameters
    ----------
    a_rec_feat : TYPE
        DESCRIPTION.
    b_rec_feat : TYPE
        DESCRIPTION.
    a_rec_data : TYPE
        DESCRIPTION.
    b_rec_data : TYPE
        DESCRIPTION.
    blockanlen : TYPE, optional
        DESCRIPTION. The default is 10000.
    cumdist : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    relen_blocks : TYPE
        DESCRIPTION.

    """
    
    
    relen_blocks=[]        
    block_lengths=[]
    frameno=0
    while frameno <= len(a_rec_data['sc-torsions']):
        if cumdist is True:
            block_lengths.append([0,frameno+blockanlen])
        else:
            block_lengths.append([frameno+1,frameno+blockanlen])
        frameno+=blockanlen
    
    for bl in block_lengths:
        print('block length = ', bl)       
        
        relen = unc_relative_entropy_analysis(a_rec_feat, b_rec_feat,
                                              a_rec_data, b_rec_data,
                                              block_length=bl, verbose=True)        
        relen_blocks.append(relen)
            
            
    np.save('relen_bl'+str(blockanlen),np.array(relen_blocks))
    
    return relen_blocks


def gradient(x,y):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    grad : TYPE
        DESCRIPTION.

    """
    x=np.array(x)
    y=np.array(y)
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]
    grad=dy/dx
    return grad

def pop_arr_val(listid, pop_val):
    """
    

    Parameters
    ----------
    listid : TYPE
        DESCRIPTION.
    pop_val : TYPE
        DESCRIPTION.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

    """
    arr = []
    for resno in range(len(listid)):
        pop_idx = np.where(listid[resno] == pop_val)
        popped = list(np.delete(listid[resno], pop_idx))
        if len(popped) > 0:
            arr.append(popped)
        else:
            arr.append([pop_val])
    return arr


def expfunc(x, a, b, c):
    """
    Create an exponential fit for an array.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.exp(a + b * np.array(x)) + c


def ssi_sem_analysis(ssi_namelist, ssi_blocks, write_plot=True, expfit=False):
    """
    

    Parameters
    ----------
    ssi_namelist : TYPE
        DESCRIPTION.
    ssi_blocks : TYPE
        DESCRIPTION.
    write_plot : TYPE, optional
        DESCRIPTION. The default is True.
    expfit : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    avsemvals : TYPE
        DESCRIPTION.
    avresssivals : TYPE
        DESCRIPTION.
    resssivals : TYPE
        DESCRIPTION.

    """
    
    ssi_dat = ssi_blocks
    ssi_names = [string[:3] for string in ssi_namelist] 
    resnames = list(set(ssi_names))
    resids = []
    
    for i in range(len(resnames)):
        resids.append(list(np.where(np.array(ssi_names) == resnames[i])[0]))
    
    resssivals = []
    avresssivals=[]
    avsemvals=[]
    
    arr = pop_arr_val(ssi_blocks, -1)
    
    for i in range(len(resids)):
        resssivals.append([arr[index] for index in resids[i]])
        avresssivals.append(list(np.average(np.array([arr[index] for index in resids[i]],dtype=object),axis=0)))
        avsemvals.append([scipy.stats.sem(avresssivals[-1][:seg]) for seg in range(len(avresssivals[-1]))])

    if write_plot is True:    
        for i in range(len(resnames)):
            
            x=list(range(len(avsemvals[i][2:])))
            y=avsemvals[i][2:]
            
            
            plt.figure()
            plt.scatter(x, y,label='raw data',marker='x',color='r')
            plt.title(resnames[i])
            plt.xlabel('Block # (in steps of 10,000 frames per simulation)')
            plt.ylabel('SSI standard error for residue type')

            ## Convergence Test
            if expfit is True:
                expofit=np.polyfit(x, np.log(y), 1)
                expy=[np.exp(expofit[0]*xval+expofit[1]) for xval in x]
                
                a=expofit[1]
                b=expofit[0]
                c=min(expy)
                p0=[a,b,c]
                popt, pcov = curve_fit(expfunc, x, y,p0=p0)
                x1 = np.linspace(min(x),200,num =1700)
                y1 = func(x1, *popt)
                plt.plot(x1, y1,  label='Exp. fit',alpha=0.75)
                plt.axhline(popt[-1],label='Converged value =~: ' +str(round(popt[-1],5)),linestyle='--',color='k')
                plt.legend()
        
            
    return avsemvals, avresssivals, resssivals

  
def relen_sem_analysis(relen_blocks, write_plot=True, expfit=False):
    """
    

    Parameters
    ----------
    relen_blocks : TYPE
        DESCRIPTION.
    write_plot : TYPE, optional
        DESCRIPTION. The default is True.
    expfit : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    resrelenvals : TYPE
        DESCRIPTION.
    avresrelenvals : TYPE
        DESCRIPTION.
    avsemvals : TYPE
        DESCRIPTION.

    """

    relen_names = [relen_dat[i][0][0][-7:-4] for i in range(len(relen_dat))]
    namesnodups = list(set(relen_names))
    
    matching_indices = []
    
    for i in namesnodups:
        matching_indices.append(list(np.where(np.array(relen_names)==i)[0]))
    
    
    resrelenvals = []
    avresrelenvals=[]
    avsemvals=[]
    
    for i in range(len(matching_indices)):
        datain=[list(relen_dat[index][1])[:-1] for index in matching_indices[i]]
        respre = []
        for sub in range(len(datain)):
            respre.append([float(val) for val in datain[sub]])
        resrelenvals.append(respre)
        avresrelenvals.append(list(np.average(np.array(resrelenvals[-1]),axis=0)))
        avsemvals.append([scipy.stats.sem(avresrelenvals[-1][:seg]) for seg in range(len(avresrelenvals[-1]))])
    
    ## Plotting the sem over each block to see the convergence
    if write_plot is True:
        for i in range(len(matching_indices)):
            
            x=list(range(len(avsemvals[i][2:])))
            y=avsemvals[i][2:]
            
            plt.figure()
            plt.scatter(x, y,label='raw data',marker='x',color='r')
            plt.title(namesnodups[i])
            plt.xlabel('Block # (in steps of 10,000 frames per simulation)')
            plt.ylabel('JSD average standard error for residue type')
            plt.savefig(namesnodups[i] + 'standarderrorrelen.png')

            if expfit is True:
                expofit=np.polyfit(x, np.log(y), 1)
                expy=[np.exp(expofit[0]*xval+expofit[1]) for xval in x]
                Wednesday, Jan
                a=expofit[1]
                b=expofit[0]
                c=min(expy)
                p0=[a,b,c]
                popt, pcov = curve_fit(expfunc, x, y,p0=p0)
                x1 = np.linspace(min(x),200,num =1700)
                y1 = func(x1, *popt)
                plt.plot(x1, y1,  label='Exp. fit',alpha=0.75)
                plt.axhline(popt[-1],label='Converged value =~: ' +str(round(popt[-1],5)),linestyle='--',color='k')
                plt.legend()                

    return resrelenvals, avresrelenvals, avsemvals


"----------------------------------------------------------------------------"
"----------------------------------------------------------------------------"
"----------------------------------------------------------------------------"


xtc1 = '../MOR_simulation/6ddf/prodmd/traj_gpcrnoh_noskip_pensa.xtc'
xtc2 = '../MOR_simulation/5c1m/prodmd/traj_gpcrnoh_noskip_pensa.xtc'

groin = '../MOR_simulation/5c1m/prodmd/gpcrnoh.gro'

start_frame=0  
a_rec = get_structure_features(groin, 
                                xtc1,
                                start_frame, features=['sc-torsions'])
a_rec_feat, a_rec_data = a_rec

b_rec = get_structure_features(groin, 
                                xtc2,
                                start_frame, features=['sc-torsions'])
b_rec_feat, b_rec_data = b_rec


relen_dat = relen_block_analysis(a_rec_feat['sc-torsions'],
                                 b_rec_feat['sc-torsions'],
                                 a_rec_data['sc-torsions'],
                                 b_rec_data['sc-torsions'], 
                                 blockanlen=10000, cumdist=False, verbose=True)
    
resrelenvals, avresrelenvals, avsemvals = relen_sem_analysis(relen_dat)

ssi_dat = ssi_block_analysis(a_rec_feat, b_rec_feat,
                             a_rec_data, b_rec_data,
                             torsions='sc', verbose=True, 
                             blockanlen=10000, cumdist=False)


avsemvals, avresssivals, resssivals = ssi_sem_analysis(ssi_names, ssi_dat)

# relen_dat = np.transpose(np.load("relen_bl10000_br10.npy"))
# ssi_dat = np.transpose(np.load("ssi_bl10000_br180.npy"))
# ssi_names = np.transpose(np.load("ssi_datanames.npy"))



