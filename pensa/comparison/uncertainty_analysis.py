import numpy as np
from pensa.features import *
from pensa.statesinfo import *
from pensa.comparison import relative_entropy_analysis
import scipy as sp
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


# -- Functions to uncertainty analysis on statistics across paired ensembles --

def _unc_ssi_ensemble_analysis(features_a, features_b, all_data_a, all_data_b, torsions=None, pocket_occupancy=None, pbc=True, 
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
    block_length : int, optional
        Length of block to be used in the block analysis. Trajectory is then
        segmented into X equal size blocks. The default is None.
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
         mv_res_feat_a, mv_res_data_a = features_a, all_data_a
         mv_res_feat_b, mv_res_data_b = features_b, all_data_b
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
                # Combine the ensembles into one distribution (condition_a + condition_b)
                data_both = list(data_a[dist_no][BL1:BL2]) + list(data_b[dist_no][BL1:BL2])      
                combined_dist.append(data_both)
            # Save distribution length
            traj1_len = len(data_a[dist_no][BL1:BL2])       
        else:
            for dist_no in range(len(data_a)):
                # Combine the ensembles into one distribution (condition_a + condition_b)
                data_both = list(data_a[dist_no]) + list(data_b[dist_no])      
                combined_dist.append(data_both)
            traj1_len = len(data_a[dist_no]) 

        # Correct for the periodicity of angles
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
                    state_limits = determine_state_limits(
                        feat_distr[dim_num], traj1_len, gauss_bins=bin_no,
                        write_plots=write_plots, write_name=plot_name
                    )
                    feat_states.append(state_limits)
                except:
                    print('Distribution A not clustering properly.')
                    print('Try altering Gaussian parameters or input custom states.')
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



def ssi_block_analysis(features_a, features_b, all_data_a, all_data_b,
                       torsions='sc', blockanlen=10000, cumdist=False, verbose=True):
    
    """
    Block analysis on the State Specific Information statistic for each feature across two ensembles.


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
        The default is 'sc'.
    blockanlen : int, optional
        Length of block to be used in the block analysis. Trajectory is then
        segmented into X equal size blocks. The default is None.
    cumdist : bool, optional
        If True, set the block analysis to a cumulative segmentation, increasing
        in length by the block length. The default is False.
    verbose : bool, default=True
        Print intermediate results.
        
    Returns
    -------
    ssi_names : list 
        Feature names of the ensembles.
    ssi_blocks : list of lists
        State Specific Information statistics for each feature, for each block.

    """
    
    # Determine the length of the trajectories
    if torsions is None:
        length_a = len(all_data_a[0][0])
        length_b = len(all_data_a[0][0])
    else:
        length_a = len(all_data_a[torsions+'-torsions'])
        length_b = len(all_data_b[torsions+'-torsions'])
    print('Trajectory lengths:', length_a, length_b)
        
    # Define the block limits
    ssi_blocks = []
    block_lengths = []
    frameno = 0
    while frameno <= min(length_a, length_b):
        if cumdist is True:
            block_lengths.append([0, frameno+blockanlen])
        else:
            block_lengths.append([frameno+1, frameno+blockanlen])
        frameno += blockanlen

    # Run the SSI analysis on each block
    for bl in block_lengths:
        print('block length = ', bl)
        ssi_names, data_ssi = _unc_ssi_ensemble_analysis(
            features_a, features_b, all_data_a, all_data_b,
            torsions=torsions, block_length=bl, verbose=True
        )
        ssi_blocks.append(data_ssi)

    ssi_names, ssi_blocks =  np.transpose(ssi_names), np.transpose(ssi_blocks)
    return ssi_names, ssi_blocks



def relen_block_analysis(features_a, features_b, all_data_a, all_data_b, 
                         blockanlen=10000, cumdist=False, verbose=True):
    """
    Block analysis on the relative entropy metrics for each feature from two ensembles.
    

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
    blockanlen : int, optional
        Length of block to be used in the block analysis. Trajectory is then
        segmented into X equal size blocks. The default is None.
    cumdist : bool, optional
        If True, set the block analysis to a cumulative segmentation, increasing
        in length by the block length. The default is False.
    verbose : bool, default=True
        Print intermediate results.

    Returns
    -------
    relen_blocks : list of lists
        List of relative entropy analysis outputs for each block.

    """
    
    relen_blocks=[]        
    block_lengths=[]
    frameno=0
    while frameno <= min(len(all_data_a),len(all_data_b)):
        if cumdist is True:
            block_lengths.append([0,frameno+blockanlen])
        else:
            block_lengths.append([frameno+1,frameno+blockanlen])
        frameno+=blockanlen
    
    for bl in block_lengths:
        block_data_a = all_data_a[bl[0]:bl[1]]
        block_data_b = all_data_b[bl[0]:bl[1]]

        print('block length = ', bl)       
        relen = relative_entropy_analysis(features_a, features_b, block_data_a, block_data_b, verbose=True)        
        relen_blocks.append(relen)
    
    return np.transpose(relen_blocks)


def _pop_arr_val(listid, pop_val):
    """
    Remove value from list. Necessary for SEM calculations which include an 
    error value of -1 for SSI. 

    Parameters
    ----------
    listid : list of lists
        State Specific Information statistics for each feature, for each block.
    pop_val : int
        Value to delete from list.

    Returns
    -------
    arr : list
        State Specific Information statistics for each feature, for each block
        with the pop_val removed.

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


def _expfunc(x, a, b, c):
    """
    Create an exponential for an x-range and exponential coefficients.

    Parameters
    ----------
    x : list
        Range of values on the x-axis.
    a : int
        Coefficient for exponential function.
    b : int
        Coefficient for exponential function.
    c : int
        Coefficient for exponential function.

    Returns
    -------
    list
        Exponential y-axis values.

    """
    return np.exp(a + b * np.array(x)) + c


def ssi_sem_analysis(ssi_namelist, ssi_blocks, write_plot=True, expfit=False, plot_dir='./SEM_plots'):
    """
    Standard error analysis for the block averages.

    Parameters
    ----------
    ssi_namelist : TYPE
        DESCRIPTION.
    ssi_blocks : TYPE
        DESCRIPTION.
    write_plots : bool, optional
        If true, visualise the SEM analysis. Default is True.
    expfit : bool, optional
        If True, apply an exponential fit to the SEM plot to predict the SEM
        value upon full convergence. Not yet fully accurate. The default is False.
    plot_dir : str, optional
        Directory in which to save the plots (if write_plots == True)
        
    Returns
    -------
    avsemvals : list of lists
        SEM values averaged across each residue type.
    avresssivals : list of lists
        SSI values for each block, averaged by residue type.
    resssivals : list of lists
        SSI values for each block, for each residue type.

    """
    
    ssi_names = [string[:3] for string in ssi_namelist] 
    resnames = list(set(ssi_names))
    resids = []
    
    for i in range(len(resnames)):
        resids.append(list(np.where(np.array(ssi_names) == resnames[i])[0]))
    
    resssivals = []
    avresssivals=[]
    avsemvals=[]
    
    arr = _pop_arr_val(ssi_blocks, -1)
    
    for i in range(len(resids)):
        resssivals.append([arr[index] for index in resids[i]])
        avresssivals.append(list(np.average(np.array([arr[index] for index in resids[i]],dtype=object),axis=0)))
        avsemvals.append([scipy.stats.sem(avresssivals[-1][:seg]) for seg in range(len(avresssivals[-1]))])

    if write_plot is True:    
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        for i in range(len(resnames)):
            
            x=list(range(len(avsemvals[i][2:])))
            y=avsemvals[i][2:]
            
            
            plt.figure()
            plt.ion()
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
                popt, pcov = curve_fit(_expfunc, x, y,p0=p0)
                x1 = np.linspace(min(x),200,num =1700)
                y1 = _expfunc(x1, *popt)
                plt.plot(x1, y1,  label='Exp. fit',alpha=0.75)
                plt.axhline(popt[-1],label='Converged value =~: ' +str(round(popt[-1],5)),linestyle='--',color='k')
                plt.legend()
            plt.ioff()                
            plt.savefig(plot_dir+'/' + resnames[i] + 'standarderrorSSI.png')
        
            
    return avsemvals, avresssivals, resssivals

  
def relen_sem_analysis(relen_dat, write_plot=True, expfit=False, plot_dir='./SEM_plots'):
    """
    Standard error analysis for the block averages.


    Parameters
    ----------
    relen_dat : list of lists
        List of relative entropy analysis outputs for each block.
    write_plots : bool, optional
        If true, visualise the SEM analysis. Default is True.
    expfit : bool, optional
        If True, apply an exponential fit to the SEM plot to predict the SEM
        value upon full convergence. Not yet fully accurate. The default is False.
    plot_dir : str, optional
        Directory in which to save the plots (if write_plots == True)

    Returns
    -------
    resrelenvals : list of lists
        JSD values for each block, for each residue type.
    avresrelenvals : list of lists
        JSD values for each block, averaged by residue type.
    avsemvals : list of lists
        SEM values averaged across each residue type.

    """

    relen_names = [relen_i[0][0].split(' ')[2] for relen_i in relen_dat]
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
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for i in range(len(namesnodups)):
            print("plotting res", i,  namesnodups[i])
            
            x=list(range(len(avsemvals[i][2:])))
            y=avsemvals[i][2:]
            
            plt.figure()      
            plt.ion()
            plt.scatter(x, y,label='raw data',marker='x',color='r')
            plt.title(namesnodups[i])
            plt.xlabel('Block # (in steps of 10,000 frames per simulation)')
            plt.ylabel('JSD average standard error for residue type')
            if expfit is True:
                expofit=np.polyfit(x, np.log(y), 1)
                expy=[np.exp(expofit[0]*xval+expofit[1]) for xval in x]
                a=expofit[1]
                b=expofit[0]
                c=min(expy)
                p0=[a,b,c]
                popt, pcov = curve_fit(_expfunc, x, y,p0=p0)
                x1 = np.linspace(min(x),200,num =1700)
                y1 = _expfunc(x1, *popt)
                plt.plot(x1, y1,  label='Exp. fit',alpha=0.75)
                plt.axhline(popt[-1],label='Converged value =~: ' +str(round(popt[-1],5)),linestyle='--',color='k')
                plt.legend()   
            plt.ioff()
            plt.savefig(plot_dir+'/' + namesnodups[i] + 'standarderrorJSD.png')

    return resrelenvals, avresrelenvals, avsemvals



## EXAMPLE USAGE


# xtc1 = 'xtc1.xtc'
# xtc2 = 'xtc2xtc'

# groin = 'gro12.gro'

# start_frame=0  
# a_rec = get_structure_features(gro1, 
#                                xtc1,
#                                start_frame, features=['sc-torsions'])
# a_rec_feat, a_rec_data = a_rec

# b_rec = get_structure_features(gro2, 
#                                xtc2,
#                                start_frame, features=['sc-torsions'])
# b_rec_feat, b_rec_data = b_rec

# relen_dat = relen_block_analysis(a_rec_feat['sc-torsions'],
#                                  b_rec_feat['sc-torsions'],
#                                  a_rec_data['sc-torsions'],
#                                  b_rec_data['sc-torsions'], 
#                                  blockanlen=1000, cumdist=False, verbose=True)
 
# resrelenvals, avresrelenvals, avsemvals = relen_sem_analysis(relen_dat)

# ssi_names, ssi_dat = ssi_block_analysis(a_rec_feat, b_rec_feat,
#                                         a_rec_data, b_rec_data,
#                                         torsions='sc', verbose=True, 
#                                         blockanlen=1000, cumdist=False)

# avsemvals, avresssivals, resssivals = ssi_sem_analysis(ssi_names, ssi_dat)
