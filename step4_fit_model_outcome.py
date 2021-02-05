from collections import Counter
import os
import pickle
import subprocess
import numpy as np
import pandas as pd
#import statsmodels.formula.api as smf
from step2_fit_model_delirium import MyCalibrator, LTRPairwise, MyLogisticRegression


if __name__=='__main__':
    model_type = 'ltr'
    random_state = 2020
    
    ## load model
    
    with open(f'results_{model_type}_Nbt0.pickle', 'rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
    
    ## load dataset
    
    with pd.ExcelFile('data_to_fit.xlsx') as xls:
        dfX = pd.read_excel(xls, 'X')
        dfy = pd.read_excel(xls, 'y')
        df_info = pd.read_excel(xls, 'info')
        df_worst_delirium_names = pd.read_excel(xls, 'worst_delirium_names')
    worst_delirium_Xnames = df_worst_delirium_names.EEGName.values.astype(str)
    dfy = dfy.rename(columns={
            'Deceased at hosp disch (0=N; 1=Y)':'DeceasedDisch',
            'Deceased at 3-mo post disch.  (0=N, 1=Y, 2=Unk)':'Deceased3month',
            'Disch. GOS':'GOSDisch'})
            
    # reverse normal EEG features
    normal_EEG_names = ['PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.',
       'Sleep patterns (Spindles, K-complex, Vertex waves)',
       'Symmetry (e.g. no focal slowing)']
    for col in normal_EEG_names:
        dfX.loc[:,col] = 1-dfX[col]
    dfX = dfX.rename(columns={col:'no '+col for col in normal_EEG_names})
    # combine no PDR and g delta/theta slowing or GRDA
    dfX['no PDR or g delta/theta slowing or GRDA'] = ((dfX['no PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.'] + dfX['Generalized/Diffuse delta or theta slowing or GRDA'])>0).astype(int)
    dfX = dfX.drop(columns=['no PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.', 'Generalized/Diffuse delta or theta slowing or GRDA'])
            
    ## for duplicate patients, pick initial visit
    
    counter = Counter(df_info.MRN)
    duplicate_mrns = [x for x in counter if counter[x]>1]
    exclude_ids = []
    for mrn in duplicate_mrns:
        ids = np.where(df_info.MRN==mrn)[0]
        init_date = df_info['Eval. date/time'].iloc[ids].min()
        exclude_ids.extend(ids[df_info['Eval. date/time'].iloc[ids]!=init_date])
    ids = ~np.in1d(np.arange(len(df_info)), exclude_ids)
    dfX = dfX[ids].reset_index(drop=True)
    dfy = dfy[ids].reset_index(drop=True)
    df_info = df_info[ids].reset_index(drop=True)
    
    dfy.loc[dfy['CAM-S LF (0-19)']>=15, 'CAM-S LF (0-19)'] = 15
    y = dfy['CAM-S LF (0-19)']
    
    assert len(set(dfX.MRN))==len(dfX)

    ## predict
    
    worst_delirium_CAMSLF = 15
    worst_delirium_VECAMS = 20
    X = dfX[Xnames].values.astype(float)
    worst_delirium_mask = np.in1d(Xnames, worst_delirium_Xnames)
    good_ids = np.all(X[:,worst_delirium_mask]!=1, axis=1)
    X = X[good_ids][:,~worst_delirium_mask]
    yp_scaled = np.zeros(len(dfX))+worst_delirium_CAMSLF
    yp_scaled[good_ids] = model.base_estimator.predict_z(X)/worst_delirium_VECAMS*worst_delirium_CAMSLF
    #yp_scaled[good_ids] = model.predict(X)
    yp = np.zeros(len(dfX))+worst_delirium_VECAMS
    yp[good_ids] = model.base_estimator.predict_z(X)
    
    ## get data
    
    ynames = ['DeceasedDisch', 'Deceased3month', 'GOSDisch']
    family = ['binomial', 'binomial', 'ordinal']
    df = dfy[ynames]
    df = df.assign(Age=df_info.Age)
    df = df.assign(Sex=(df_info.Gender=='M').astype(float))
    df = df.assign(**{'CAMSLF':y})
    df = df.assign(**{'Scaled_VECAMS_0_15':yp_scaled})
    df = df.assign(**{'VECAMS':yp})
    df.to_csv('step4_output_df.csv', index=False)
    Xnames = ['Age', 'Sex', 'CAMSLF', 'Scaled_VECAMS_0_15']
    
    # normalize
    for xn in ['Age']:
        df[xn] = (df[xn] - df[xn].mean() ) / df[xn].std()
    
    cwd = os.getcwd()
    Xpath = os.path.join(cwd, 'data.csv')
    code_path = os.path.join(cwd, 'tmp.R')
    for yi, yname in enumerate(ynames):
        result_path = os.path.join(cwd, f'coef_model_outcome_{yname}.csv')
    
        # save input
        if yname=='Deceased3month':
            goodids = np.in1d(df[yname], [0,1])
            df.iloc[goodids].to_csv(Xpath, index=False)
        else:
            df.to_csv(Xpath, index=False)
            
        # save R code
        rcode = f'''mydata <- read.csv("{Xpath}")
'''
        if family[yi] == 'binomial':
            rcode += f'''mdl <- glm({yname} ~ {'+'.join(Xnames)}, data=mydata, family="binomial")
coefs <- coef(summary(mdl))
cis <- confint(mdl)
res <- cbind(coefs, cis)
'''
        elif family[yi] == 'ordinal':
            rcode += f'''library(MASS)
mdl <- polr(as.factor({yname}) ~{'+'.join(Xnames)}, data=mydata, Hess=TRUE)
coefs <- coef(summary(mdl))
cis <- confint(mdl)
pvals <- pnorm(abs(coefs[, "t value"]), lower.tail = FALSE) * 2
coefs <- cbind(coefs, "Pr(>|z|)" = pvals)[c("{'","'.join(Xnames)}"),]
res <- cbind(coefs, cis)
'''
        rcode += f'write.csv(res, "{result_path}")'
        
        with open(code_path, 'w') as ff:
            ff.write(rcode)
        
        # run R code
        subprocess.check_call(['Rscript', code_path])
        
        # load result
        #df_res = pd.read_csv(result_path)
        
        # delete tmp files
        os.remove(code_path)        
        os.remove(Xpath)
        
