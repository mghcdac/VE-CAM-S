import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress, spearmanr
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set_style('ticks')


def bootstrap_curves(x, xs, ys, verbose=True):
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    idx = np.argsort(x)
    x_res = x[idx]
    
    ys_res = []
    for _ in tqdm(range(len(xs)), disable=not verbose):
        try:
            xx = xs[_]
            yy = ys[_]
            _, idx = np.unique(xx, return_index=True)
            idx = np.sort(idx)
            xx = xx[idx]; yy = yy[idx]
            idx = np.argsort(xx)
            xx = xx[idx]; yy = yy[idx]
            foo = interp1d(xx, yy, kind='cubic')
            ys_res.append( foo(x) )
        except Exception as ee:
            print(str(ee))
            continue
    ys_res = np.array(ys_res)
    
    return x_res, ys_res
    

if __name__ == '__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
        
    K = 16
    model_type = 'ltr'
    df_pred = pd.read_csv(f'../cv_predictions_{model_type}_Nbt0.csv')
    df_pred['SID'] = df_pred.SID.astype(str)
    df_pred = df_pred[(df_pred.cvi!='full')&(df_pred.bti==0)].reset_index(drop=True)
    y = df_pred.y.values
    yp = df_pred.z.values

    Nbt = 1000
    random_state = 2020
    np.random.seed(random_state)
    ys = []; yps = []; yp_probs = []
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            df_bt = df_pred.copy()
        else:
            btids = np.random.choice(len(df_pred), len(df_pred), replace=True)
            df_bt = df_pred.iloc[btids].reset_index(drop=True)
        ys.append( df_bt.y.values )
        yps.append( df_bt.z.values )
        yp_probs.append( df_bt[[f'prob({x})' for x in range(K)]].values )
        #yps_int.append( np.argmax(yp_probs, axis=1) )
    corrs = [spearmanr(ys[i], yps[i])[0] for i in range(len(ys))]
    corr = corrs[0]
    corr_lb, corr_ub = np.percentile(corrs, (2.5, 97.5))
    
    panel_xoffset = -0.12
    panel_yoffset = 1.01
    
    figsize = (11,10)
    # scatter plot
    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    bp = ax.boxplot([yp[y==i] for i in range(K)], positions=np.arange(K))
    plt.setp(bp['medians'], color='r', lw=2)
    ax.scatter(y+np.random.randn(len(y))/20., yp+np.random.randn(len(yp))/5., s=40, ec='none', alpha=0.2, fc='k')
    ax.text(0.03, 0.91, f'Spearman\'s correlation R = {corr:.2f} ({corr_lb:.2f} -- {corr_ub:.2f})',
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('CAM-S LF score')
    ax.set_ylabel('VE-CAM-S score')
    ax.set_xlim([-1,16])
    ax.set_ylim([-1,21])
    ax.yaxis.grid(True)
    sns.despine()
    ax.text(panel_xoffset/2, panel_yoffset, 'A', ha='right', va='top', transform=ax.transAxes, fontweight='bold')

    """
    # confusion matrix plot
    cf = confusion_matrix(ys, yps_int)
    plt.close()
    fig=plt.figure(figsize=(12,9))
    ax=fig.add_subplot(111)
    sns.heatmap(np.flipud(cf.T),annot=True,cmap='Blues',fmt='d')
    ax.set_yticklabels(np.arange(K)[::-1])
    ax.set_ylabel('Predicted CAM-S')
    ax.set_xlabel('Actual CAM-S')
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'confusionmatrix_{model_type}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'confusionmatrix_{model_type}.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    """

    # AUC
    ax = fig.add_subplot(gs[1,0])
    #levels = np.arange(K-1)
    levels = [4]
    for i in levels:
        y2 = [(ys[bti]>i).astype(int) for bti in range(Nbt+1)]
        yp2 = [yp_probs[bti][:, i+1:].sum(axis=1) for bti in range(Nbt+1)]
        aucs = [roc_auc_score(y2[bti], yp2[bti]) for bti in range(Nbt+1)]
        fprs_tprs = [roc_curve(y2[bti], yp2[bti]) for bti in range(Nbt+1)]
        fprs = [x[0] for x in fprs_tprs]
        tprs = [x[1] for x in fprs_tprs]
        fpr, tprs = bootstrap_curves(fprs[0], fprs, tprs)
        auc = aucs[0]
        tpr = tprs[0]
        if Nbt>0:
            auc_lb, auc_ub = np.percentile(aucs[1:], (2.5, 97.5))
            tpr_lb, tpr_ub = np.percentile(tprs[1:], (2.5, 97.5), axis=0)
        else:
            auc_lb = np.nan
            auc_ub = np.nan
        if Nbt>0:
            ax.fill_between(fpr, tpr_lb, tpr_ub, color='k', alpha=0.2, label='95% CI')
        ax.plot(fpr, tpr, c='k', lw=2, label=f'CAM-S LF <= {i} vs. >={i+1}:\nAUC = {auc:.2f} [{auc_lb:.2f} - {auc_ub:.2f}]')# (n={np.sum(y2[0]==0)})
        
    ax.plot([0,1],[0,1],c='k',ls='--')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.grid(True)
    sns.despine()
    ax.text(panel_xoffset, panel_yoffset, 'B', ha='right', va='top', transform=ax.transAxes, fontweight='bold')

    # calibration
    ax = fig.add_subplot(gs[1,1])
    ax.plot([0,1],[0,1],c='k',ls='--')
    #levels = np.arange(K-1)
    levels = [4]
    for i in levels:
        y2 = [(ys[bti]>i).astype(int) for bti in range(Nbt+1)]
        yp2 = [yp_probs[bti][:, i+1:].sum(axis=1) for bti in range(Nbt+1)]
        obss_preds = [calibration_curve(y2[bti], yp2[bti], n_bins=10, strategy='quantile') for bti in range(Nbt+1)]
        obss = [x[0] for x in obss_preds]
        preds = [x[1] for x in obss_preds]
        pred, obss = bootstrap_curves(preds[0], preds, obss)
        obs = obss[0]
        cslopes = [linregress(x[1], x[0])[0] for x in obss_preds]
        cslope, intercept, _, _, _ = linregress(pred, obs)
        if Nbt>0:
            cslope_lb, cslope_ub = np.percentile(cslopes[1:], (2.5, 97.5))
            obs_lb, obs_ub = np.percentile(obss[1:], (2.5, 97.5), axis=0)
        else:
            cslope_lb = np.nan
            cslope_ub = np.nan
        if Nbt>0:
            ax.fill_between(pred, obs_lb, obs_ub, color='k', alpha=0.2, label='95% CI')
        ax.plot(pred, cslope*pred+intercept, c='k', lw=2, label=f'CAM-S LF <= {i} vs. >={i+1}:\ncalib. slope = {cslope:.2f} [{cslope_lb:.2f} - {cslope_ub:.2f}]')# (n={np.sum(y2[0]==0)})
        ax.scatter(pred, obs, c='k', marker='o', s=40)
        
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed fraction')
    ax.grid(True)
    ax.text(panel_xoffset, panel_yoffset, 'C', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
    sns.despine()
    
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'performance_{model_type}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'performance_{model_type}.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
