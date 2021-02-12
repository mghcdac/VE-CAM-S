import numpy as np
import pandas as pd
from collections import Counter

## read mastersheet

#df = pd.read_excel('mastersheet_reformatted.xlsx', sheet_name='Dataset_full')
df = pd.read_excel('mastersheet_combined_reformatted.xlsx', sheet_name='Dataset_full')
# rename columns
df = df.rename(columns={k:k.strip() for k in df.columns})
df = df.rename(columns={'Total Points (Add age score to comorbidy score)' : 'CCI'})
# exclude two patients without EEG
df = df[(df.SID!='AMSD086') & (df.SID!='AMSD153')].reset_index(drop=True)
df['Eval. date/time'] = pd.to_datetime(df['Eval. date/time'])

## generate sids

df['MRN'] = df.MRN.astype(str)
two_mrn_ids = np.where(df.MRN.str.contains('\('))[0]
df.loc[two_mrn_ids, 'MRN'] = df.MRN.iloc[two_mrn_ids].str.replace('(','_').str.replace(')','').str.split('_', expand=True)[1]
underscore_ids = np.where(df.MRN.str.contains('_'))[0]
df.loc[underscore_ids, 'MRN'] = df.MRN.iloc[underscore_ids].str.replace('_','')
assert np.all(df.MRN.str.isnumeric())
sid_names = ['SID', 'MRN']

"""
## generate demographics

df['Gender'] = df.Gender.str.strip().str.upper()
df.loc[df.Gender=='M', 'Gender'] = 1
df.loc[df.Gender=='F', 'Gender'] = 0
df['Age>=50'] = (df.Age>=50).astype(int)
df['Age>=60'] = (df.Age>=60).astype(int)
df['Age>=70'] = (df.Age>=70).astype(int)
df['Age>=80'] = (df.Age>=80).astype(int)
demo_name = []#'Age>=50', 'Age>=60', 'Age>=70', 'Age>=80', 'Gender']#, 'CCI']
demo = df[demo_name].values.astype(float)
"""

## define eeg names

eeg_names = [
'PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.',
'Sleep patterns (Spindles, K-complex, Vertex waves)',
'Symmetry (e.g. no focal slowing)',

'Generalized/Diffuse delta slowing',
'Generalized/Diffuse theta slowing',
#'Diffuse slowing - Either theta or delta or GRDA',
'Excess/Diffuse alpha',
#'Excess/Diffuse beta',
'Focal/Unilateral delta slowing',
'Focal/Unilateral theta slowing',
#'Focal slowing - Either theta or delta or LRDA',
'GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)',
'LRDA (Lateralized rhythmic delta activity)',
'Extreme delta brush', ##

#'Periodic discharges - LPD or GPD or BiPD',
#'Any IIIC: LPD or GPD or BiPD or LRDA or Sz or NCSE or TPW (TPW or GPD with TP morphology)',
'LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)',

'GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)',
'GPD with Triphasic morphology',
'Triphasic waves',

'Sporadic epileptiform discharges (=sporadic discharges)',
'BIPD (bilateral indep. periodic discharges) (=BIPLED - Bilateral independent periodic lateralized epileptiform discharges)', ##
'BIRDs (brief potentially ictal rhythmic discharges)',

'Discrete seizures: generalized', 
#'Discrete seizures: focal',
'Non convulsive status epilepticus: generalized', ##
'Non convulsive status epilepticus: focal',

'Burst suppression with epileptiform activity', ##
'Burst suppression without epileptiform activity', ##
'Intermittent brief attenuation',
'Moderately low voltage',
'Extremely low voltage / electrocerebral silence', ##
'EEG Unreactive',] ##

# remove [] in EEG features
for col in eeg_names:
    bracket_ids = df[col].astype(str).str.contains('\[')
    if np.any(bracket_ids):
        df.loc[bracket_ids, col] = df[col][bracket_ids].str.replace(']','').str.split('[',expand=True)[0].str.replace('+','').astype(float)


# GPDnTPW/GPDTPW/TPW --> GPD
eeg_names.remove('GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)')
eeg_names.remove('GPD with Triphasic morphology')
eeg_names.remove('Triphasic waves')
eeg_names.append('GPD')
df['GPD'] = ((df['GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)'] + df['GPD with Triphasic morphology'] + df['Triphasic waves'])>0).astype(float)
df = df.drop(columns=['GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)'])
df = df.drop(columns=['GPD with Triphasic morphology'])
df = df.drop(columns=['Triphasic waves'])

# combine G delta slowing, G theta slowing, and GRDA
eeg_names.remove('Generalized/Diffuse delta slowing')
eeg_names.remove('Generalized/Diffuse theta slowing')
eeg_names.remove('GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)')
eeg_names.append('Generalized/Diffuse delta or theta slowing or GRDA')
df['Generalized/Diffuse delta or theta slowing or GRDA'] = ((df['Generalized/Diffuse delta slowing'] + df['Generalized/Diffuse delta slowing'] + df['GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)'])>0).astype(float)
df = df.drop(columns=['Generalized/Diffuse delta slowing'])
df = df.drop(columns=['Generalized/Diffuse theta slowing'])
df = df.drop(columns=['GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)'])

worst_delirium_names = [
'Extreme delta brush',
'BIPD (bilateral indep. periodic discharges) (=BIPLED - Bilateral independent periodic lateralized epileptiform discharges)',
'Non convulsive status epilepticus: generalized',
'Extremely low voltage / electrocerebral silence',
'Burst suppression with epileptiform activity',
'Burst suppression without epileptiform activity',
'EEG Unreactive',
]


"""
# combine eeg names
df['LPD or GPD or GPD with TP morphology or TPW or BIRDS'] =\
    ( (df['LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)']==1) |\
      (df['GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)']==1) |\
      (df['GPD with Triphasic morphology']==1) |\
      (df['Triphasic waves']==1) |\
      (df['BIRDs (brief potentially ictal rhythmic discharges)']==1)
    ).astype(float)
df['LPD or LRDA'] =\
    ( (df['LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)']==1) |\
      (df['LRDA (Lateralized rhythmic delta activity)']==1)
    ).astype(float)
df['attenuation'] =\
    ( (df['Intermittent brief attenuation']==1) |\
      (df['Moderately low voltage / generalized attenuation / generalized suppression']==1)
    ).astype(float)

cols_del = ['LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)',
    'GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)',
    'GPD with Triphasic morphology',
    'Triphasic waves',
    'BIRDs (brief potentially ictal rhythmic discharges)',
    'LRDA (Lateralized rhythmic delta activity)',
    'Discrete seizures: generalized',
    'Discrete seizures: focal',
    'Non convulsive status epilepticus: focal',
    'Intermittent brief attenuation',
    'Moderately low voltage / generalized attenuation / generalized suppression']
for col in cols_del:
    eeg_names.remove(col)
df = df.drop(columns=cols_del)

cols_add = [
    'LPD or GPD or GPD with TP morphology or TPW or BIRDS',
    'LPD or GPD or GPD with TP morphology or TPW or BIRDS or LRDA',
    'GPD or GPD with TP morphology or TPW',
    'LPD or LRDA',
    'Sz or NCSE',
    'attenuation']
for col in cols_add:
    eeg_names.append(col)
"""

## generate y names (label)

ynames = ['CAM-ICU (0/1)', 'CAM-S SF (0-7)', 'CAM-S LF (0-19)', 'Prorated LF Score', '3D-CAM (0/1)', '3D-CAM-S SF (0-7)', 'LOS (Days)', 'Deceased at hosp disch (0=N; 1=Y)', 'Deceased at 3-mo post disch.  (0=N, 1=Y, 2=Unk)', 'Disch. GOS', 'Disch. GOSE']

col = 'Prorated LF Score'
ids = df[col].astype(str).str.contains('/')
tmp = df[col][ids].str.split('/',expand=True)
tmp0 = tmp[0].str.replace('_','').values.astype(float)
tmp1 = tmp[1].str.replace('_','').values.astype(float)
df.loc[ids, col] = tmp0/tmp1

col = 'Deceased at 3-mo post disch.  (0=N, 1=Y, 2=Unk)'
ids = df[col].astype(str).str.contains('\(likely')
tmp = df[col][ids].str.replace(')','').str.split('\(likely',expand=True)
df.loc[ids, col] = tmp[1].astype(float)

## generate info

info_names = ['Age', 'Gender', 'Pre-hosp. GOS', 'Pre-hosp. GOSE', 'Type of EEG (routine, LTM)', 'Eval. within epoch (Y=1, N=0)', 'Eval. date/time']

df['Gender'] = df.Gender.astype(str).str.strip().str.upper()

col = 'Pre-hosp. GOS'
ids = df[col].astype(str)=='_'
df.loc[ids, col] = np.nan

col = 'Pre-hosp. GOSE'
ids = (df[col].astype(str)=='_') | (df[col].astype(str)=='?')
df.loc[ids, col] = np.nan

col = 'Type of EEG (routine, LTM)'
df[col] = df[col].str.strip().str.upper()
df.loc[df[col]=='LTM',col] = 1
df.loc[df[col]=='ROUTINE',col] = 0

col = 'Eval. within epoch (Y=1, N=0)'
ids = df[col].astype(str).str.contains('\?')
df.loc[ids, col] = df[col][ids].str.replace('?','')

counts = [df[x].sum() for x in eeg_names]
df_count = pd.DataFrame(data={'EEGName':eeg_names, 'Count':counts})
df_count = df_count.sort_values('Count', ascending=False).reset_index(drop=True)

# save
import pdb;pdb.set_trace()
with pd.ExcelWriter('data_to_fit.xlsx') as writer:  
    df[sid_names+eeg_names].to_excel(writer, sheet_name='X', index=False)
    df[sid_names+ynames].to_excel(writer, sheet_name='y', index=False)
    df[sid_names+info_names].to_excel(writer, sheet_name='info', index=False)
    
    pd.DataFrame(data={'EEGName':worst_delirium_names}).to_excel(writer, sheet_name='worst_delirium_names', index=False)
    df_count.to_excel(writer, sheet_name='counts', index=False)

