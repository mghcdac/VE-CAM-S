import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
seaborn.set_style('ticks')


def cluster_corr(corr_array):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    return corr_array[idx, :][:, idx], idx


def binary_correlation(x, y):
    """
    Correlation for binary vectors (0 or 1)
    Ref:
    [1] Zhang, B. and Srihari, S.N., 2003, September. Properties of binary vector dissimilarity measures. In Proc. JCIS Int'l Conf. Computer Vision, Pattern Recognition, and Image Processing (Vol. 1). https://cedar.buffalo.edu/papers/articles/CVPRIP03_propbina.pdf
    [2] Tubbs, J.D., 1989. A note on binary template matching. Pattern Recognition, 22(4), pp.359-365.
    """
    #corr = np.mean(X[:,i]==X[:,j])*2-1
    #corrpearsonr(X[:,i], X[:,j])[0]
    
    s11 = np.sum(x*y)
    s10 = np.sum(x*(1-y))
    s01 = np.sum((1-x)*y)
    s00 = np.sum((1-x)*(1-y))
    
    # The "Correlation" method in Table 1 of [1].
    sigma = np.sqrt( (s10+s11)*(s01+s00)*(s11+s01)*(s00+s10) )
    corr = (s11*s00-s10*s01)/sigma
    
    """
    # The "Rogers-Tanmot" method in Table 1 of [1].
    corr = (s11+s00)/(s11+s00+2*s10+2*s01)
    corr = corr*2-1  # turn it into -1 to 1
    """
    
    return corr
    
    
if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
    
    df = pd.read_excel('../data_to_fit.xlsx', sheet_name='X')
    eeg_names = list(df.columns)
    eeg_names.remove('SID')
    eeg_names.remove('MRN')
    eeg_names = np.array(eeg_names)

    name2shortname = {
        'PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.':'PDR',
        'Sleep patterns (Spindles, K-complex, Vertex waves)':'Sleep',
        'Symmetry (e.g. no focal slowing)':'Symmetry',
        
        'Generalized/Diffuse delta slowing':'G delta slowing',
        'Generalized/Diffuse theta slowing':'G theta slowing',
        'Generalized/Diffuse delta or theta slowing or GRDA':'GRDA/G delta/theta slow',
        'Diffuse slowing - Either theta or delta or GRDA':'Diffuse slowing',
        'Excess/Diffuse alpha':'Excess alpha',
        'Excess/Diffuse beta':'Excess beta',
        'Focal/Unilateral delta slowing':'F delta slowing',
        'Focal/Unilateral theta slowing':'F theta slowing',
        'Focal slowing - Either theta or delta or LRDA':'Focal slowing',
        'GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)':'GRDA',
        'LRDA (Lateralized rhythmic delta activity)':'LRDA',
        'Extreme delta brush':'EDB',

        'Periodic discharges - LPD or GPD or BiPD':'LPD/GPD/BiPD',
        'Any IIIC: LPD or GPD or BiPD or LRDA or Sz or NCSE or TPW (TPW or GPD with TP morphology)':'Any IIIC',
        'LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)':'LPD',
        'GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)':'GPD w/o TPW',
        'GPD with Triphasic morphology':'GPD w TPW',
        'Triphasic waves':'TPW',
        'GPD':'GPD',
        'Sporadic epileptiform discharges (=sporadic discharges)':'Sporadic discharges',
        'BIPD (bilateral indep. periodic discharges) (=BIPLED - Bilateral independent periodic lateralized epileptiform discharges)':'BIPD',
        'BIRDs (brief potentially ictal rhythmic discharges)':'BIRDs',

        'Discrete seizures: generalized':'G Sz', 
        'Discrete seizures: focal':'F Sz',
        'Non convulsive status epilepticus: generalized':'G NCSE',
        'Non convulsive status epilepticus: focal':'F NCSE',

        'Burst suppression with epileptiform activity':'BS w spike',
        'Burst suppression without epileptiform activity':'BS w/o spike',
        'Intermittent brief attenuation':'IBA',
        'Moderately low voltage':'MLV',
        'Extremely low voltage / electrocerebral silence':'ELV',
        'EEG Unreactive':'Unreactive'}
    
    X = df[eeg_names].values.astype(float)
    corrX = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            corrX[i,j] = binary_correlation(X[:,i], X[:,j])
    corrX2, idx = cluster_corr(corrX)
    
    cmap = 'coolwarm'
    #figsize = (17,8.2)
    figsize = (9.6,8.5)
    plt.close()
    fig = plt.figure(figsize=figsize)
    
    """
    ax = fig.add_subplot(121)
    ax.imshow(corrX, aspect='auto', vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(corrX)))
    ax.set_xticklabels([name2shortname[x] for x in eeg_names], ha='left', rotation=70)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(np.arange(len(corrX)))
    ax.set_yticklabels([name2shortname[x] for x in eeg_names])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cax.axis('off')
    """
    
    #ax = fig.add_subplot(122)
    ax = fig.add_subplot(111)
    im = ax.imshow(corrX2, aspect='auto', vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(corrX2)))
    ax.set_xticklabels([name2shortname[x] for x in eeg_names[idx]], ha='left', rotation=70)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(np.arange(len(corrX2)))
    ax.set_yticklabels([name2shortname[x] for x in eeg_names[idx]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax)#, orientation='horizontal')
    cbar.ax.set_ylabel('Correlation')
    
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig('corr_mat.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('corr_mat.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
