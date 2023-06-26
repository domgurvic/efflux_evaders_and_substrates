import sys
import os

# sys.path.append("/homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/")
# sys.path.append("~/test4/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/")

import pandas as pd
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import numpy as np
from rdkit import Chem
import re
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import statsmodels.api as sm
import pickle
from tqdm import trange, tqdm
from scipy import stats, ndimage

# tsne

from sklearn.manifold import TSNE
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Descriptors3D, Draw, rdMolDescriptors, Draw, PandasTools
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat, GetTanimotoDistMat
from rdkit.Chem.rdmolops import FastFindRings



def add_nitrogen_charges(smiles):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m

def mols_to_NHA(x): # input mmpa dataframe, grab smarts/smiles output number of heavy atoms
    return Chem.MolFromSmarts(x).GetNumHeavyAtoms()

def clean_mmpa_pairs_len(mmpa_df):
    temp=pd.DataFrame() # temp dataframe
    if 'LHS' not in mmpa_df.columns: # add LHS and RHS if not present
        mmpa_df = master_functions.split_transition(mmpa_df, 'smirks')     # produce LHS and RHS
    else:
        temp['common_core_HA'] = mmpa_df['common_core'].apply(mols_to_NHA) # produce number of heavy atoms
        temp['LHS_HA'] = mmpa_df['LHS'].apply(mols_to_NHA)
        temp['RHS_HA'] = mmpa_df['LHS'].apply(mols_to_NHA)
        
        temp['len_check'] = np.where((temp['LHS_HA'] >= temp['common_core_HA']) & (temp['RHS_HA'] >= temp['common_core_HA'])
                     , 'fail', 'pass') # compare lengths of heavy atoms
        
        mmpa_df = mmpa_df.drop(temp[temp['len_check']=='fail'].index) # drop index that failed length check
        
        print('Initial number of transofrms: {} \nNumber fo transforms disqualified based on length discrepancy: {} \nRemaining number of transforms: {}'.format(len(temp[temp['len_check']=='fail']) +  len(mmpa_df) , len(temp[temp['len_check']=='fail']), len(mmpa_df)))
        # return temp to debug
    return mmpa_df # retun mmpa dataframe without len_check == fail


def split_transition(df, col):
    df['LHS'] = [re.split('>>',df[col].values[i])[0] for i in range(len(df)) ]
    df['RHS'] = [re.split('>>',df[col].values[i])[1] for i in range(len(df)) ]
    return df


def stat_it_2(it):
    
    hts_pairs = it
    hts_smirk_repearts = hts_pairs.smirks.value_counts().index.tolist()
    hts_smirk_repearts_values = hts_pairs.smirks.value_counts().tolist()

    repeats_filtered = hts_smirk_repearts
    repeats_filtered=[y for x, y in zip(hts_smirk_repearts_values, hts_smirk_repearts) if x >= 3] # at least 3 pairs
    
    res_arr=[]
    
    print('Number of unique transforms: {} \nProcessing transforms:...\n '.format(len(repeats_filtered)))

    for i in range(len(repeats_filtered)):

        pair_eg = hts_pairs[hts_pairs['smirks'] == repeats_filtered[i]].reset_index(drop=True)

        t, p_2 = stats.ttest_rel(pair_eg.measurement_B,pair_eg.measurement_A) #Two-sided p-value / need only one side?

        new_row = {'smirks':hts_smirk_repearts[i], 'dof':len(pair_eg)-1,'t-stat':t, \
                   'p-val (t-test)':p_2, 'measurement_delta':pair_eg.measurement_B.mean() - pair_eg.measurement_A.mean(), 'std':pair_eg.measurement_delta.std(),  'sem':pair_eg.measurement_delta.sem(), }

        res_arr.append(new_row)
        
        if i % 1000 == 0 and i != 0:
            print(i)
    print('done!')
    return pd.DataFrame(res_arr, columns = ['smirks', 'dof' ,'t-stat', 'p-val (t-test)', 'measurement_delta', 'std',  'sem'])


def zero_in(dataframe,cutoff, pos_only=True):
    
    res = dataframe
    
    if pos_only==True:
        res_t_pos = res[res['t-stat']>0].reset_index(drop=True) # positive change
    else:
        res_t_pos=res
        
    res_t_pos_p_pos = res_t_pos[res_t_pos['p-val (t-test)']<cutoff] # significant change
    
    res_pos_t_sorted = res_t_pos_p_pos.sort_values(by='measurement_delta', ascending=False).reset_index(drop=True)
    
    print('Number of unique transforms where p-val < {} is {}'.format(cutoff, len(res_pos_t_sorted)))
    print('Split between {} positive transforms and {} negative transforms'.format(len(res_t_pos_p_pos[res_t_pos_p_pos['t-stat'] > 0]), len(res_t_pos_p_pos[res_t_pos_p_pos['t-stat'] < 0])))
    
    return res_pos_t_sorted




def calcualte_features(df):
    
    print('must be present with LHS and RHS, avg_pred_diff, transition')
    
    generator = MakeGenerator(("rdkit2d",))
    names=[name[0] for name in  generator.GetColumns()]
    
    l_feat=[]
    r_feat=[]
    act=[]
    smirk=[]
    
    print('Computing features: ')
    
    for i in trange(len(df.LHS.values)):

        l_data = generator.process(df.LHS.values[i])
        r_data = generator.process(df.RHS.values[i])
        try:
            if l_data[0] and r_data[0] == True:         # this???
                l_feat.append(l_data[1:])
                r_feat.append(r_data[1:])
                act.append(df['avg_pred_diff'].values[i])
                smirk.append(df['transition'].values[i])
            else:
                print('left: ', l_data[0])
                print('right: ', r_data[0])
                print(df.LHS.values[i], df.RHS.values[i])
        except TypeError:
            print('error', i)
    

    # calculate descriptor difference
    diff_feat=[]
    for k, y in zip(r_feat, l_feat):
        diff_feat.append(np.array(k)-np.array(y))

    # add descriptors to existing dataframe 
    feats = pd.DataFrame()
    print('Computing feature differences: ')
    
    for i in range(len(diff_feat)):
        feats = feats.append(pd.Series(diff_feat[i]), ignore_index=True)

    feats.columns = names[1:]
    
    print('features: ', len(feats))
    print('activities: ' ,len(act))
    print('must match')
    
    
    feats['target'] = act
    
    feats['smirks'] = smirk
    
    l_feats = pd.DataFrame()
    
    print('Adding LHS features: ')
    
    for i in range(len(l_feat)):
        l_feats = l_feats.append(pd.Series(l_feat[i]), ignore_index=True)

    l_feats.columns = names[1:]
    
    l_feats['target'] = act
    
    l_feats['smirks'] = smirk
    
    print('Adding RHS features: ')
    
    r_feats = pd.DataFrame()
    for i in range(len(r_feat)):
        r_feats = r_feats.append(pd.Series(r_feat[i]), ignore_index=True)

    r_feats.columns = names[1:]
    
    r_feats['target'] = act
    
    r_feats['smirks'] = smirk
    
    # Keep only descriptors of values without 0
    feats_2 = feats.loc[:, (feats != 0).any(axis=0)]
    
    return feats, l_feats, r_feats


def calcualte_features_single(df, col):
    
        
    generator = MakeGenerator(("rdkit2d",))
    names=[name[0] for name in  generator.GetColumns()]
    
    l_feat=[]
    
    print('Computing features: ')
    
    for i in trange(len(df[col].values)):
        l_data = generator.process(df[col].values[i])
        
        if l_data[0]  == True:
            l_feat.append(l_data[1:])
        else:
            print('left: ', l_data[0])
            print(df[col].values[i])
    

    # add descriptors to existing dataframe 
    feats = pd.DataFrame()
    
    for i in trange(len(l_feat)):
        feats = feats.append(pd.Series(l_feat[i]), ignore_index=True)

    feats.columns = names[1:]

    return feats



def find_sig_feats(l_feats, r_feats, p_val):
    
    df_sig_feats=pd.DataFrame(columns=l_feats.columns)

    #df_feats_fr_only=df_feats.iloc[:,114:199]

    for name in df_sig_feats.columns[:-2]:
        t, p = stats.ttest_rel(r_feats[name],l_feats[name])
        if p < p_val:
            df_sig_feats[name]=pd.Series([p, t])
        df_sig_feats = df_sig_feats.dropna(axis=1)

    fr_descriptors=[]
    ph_descriptors=[]
    
    for descriptor in df_sig_feats.columns:
        if re.search('fr_(.*?)', descriptor) is not None:
            fr_descriptors.append(descriptor)
        else:
            ph_descriptors.append(descriptor)
    print('Significant descriptos of fractions (fr_*) found: ', len(fr_descriptors))
    
    print('Significant descriptos of physcial features found: ', len(ph_descriptors))    
    return fr_descriptors, ph_descriptors


def plot_feats(res_neg):

    sns.set(rc={'figure.figsize':(10, 10)})

    sns.set_style("darkgrid")
    #sns.set_style("ticks")

    hist, ax = plt.subplots()

    bar = sns.barplot(data = res_neg, x=r'$\overline{\Delta P}$',y ='Main fraction', hue='Correlation', palette=["C1", "C0"])



    plt.xlabel( r'$\overline{\Delta P}$', size = 15 )

    plt.ylabel( "Main Fractions" , size = 15 )

    plt.title( "Unique Significant Transforms" , size = 18 );

    plt.tick_params(labelsize=14)

    sns.despine()


def results_arr(features_all, fr_sig_descriptors, r_feats, l_feats , fractions_to_drop):
    
    arr=[]

    #features_all_fr_only = features_all.iloc[:,114:199]
    #features_all_fr_only_target = features_all.iloc[:,114:200]


    #fr_descriptors=[]
    #for descriptor in features_all.columns:
    #    if re.search('fr_(.*?)', descriptor) is not None:
    #        fr_descriptors.append(descriptor)
    #

    #features_all_fr_only=features_all[fr_descriptors]
    
    features_all_fr_only = features_all.iloc[:,:-4]

    for name in fr_sig_descriptors: 
        if r_feats[name].mean() - l_feats[name].mean() == 0:
            print('{} has neutral correlation '.format(name))


        elif r_feats[name].mean() - l_feats[name].mean() < 0:

            print('{} has negative correlation '.format(name))


            name_loss= features_all_fr_only[features_all_fr_only[name]<0] # find all negative occurances of this named fraction

            name_loss = name_loss.drop(fractions_to_drop, axis = 1)

            # in all fractions that name is lost who are biggest gains:
            big_gain = pd.DataFrame(name_loss.mean().sort_values(ascending=False)).T.columns

            delta_p_p = round(features_all.iloc[name_loss.index].target.mean(), 2)
            
            delta_p_p_sem = round(features_all.iloc[name_loss.index].target.sem(), 2)
            
            delta_p_p_std = round(features_all.iloc[name_loss.index].target.std(), 2)
            
            #percentage_loss=[round(len(name_loss[name_loss[x]>0])/ len(name_loss) *100, 2 ) for x in big_gain[:10]]

            percentage_loss = [round(name_loss[x].sum() / len(name_loss) *100, 2 ) for x in big_gain[:10]]



            # Look at overlaping percentages

            if percentage_loss[1] == percentage_loss[2] == percentage_loss[3]:
                    big_gain=[big_gain[0], (big_gain[1] , big_gain[2], big_gain[3]), big_gain[4] ]
                    percentage_loss = [percentage_loss[0] , percentage_loss[2], percentage_loss[4]]

            elif percentage_loss[0] == percentage_loss[1] == percentage_loss[2]:
                big_gain=[( big_gain[0], big_gain[1] , big_gain[2]) , big_gain[3], big_gain[4]]
                percentage_loss = [percentage_loss[0] , percentage_loss[3], percentage_loss[4]]


                print('all gain')
                print(big_gain)
                print(percentage_loss)

            elif int(percentage_loss[0]) == int(percentage_loss[1]):
                big_gain=[(big_gain[0], big_gain[1]), big_gain[2], big_gain[3]]
                percentage_loss = [percentage_loss[0] , percentage_loss[2], percentage_loss[3]]

                print('first_gain')
                print(big_gain)
                print(percentage_loss)

            elif int(percentage_loss[1]) == int(percentage_loss[2]):
                big_gain=[big_gain[0], (big_gain[1] , big_gain[2]) , big_gain[3]]
                percentage_loss = [percentage_loss[0] , percentage_loss[2], percentage_loss[3]]


                print('second gain')
                print(big_gain)
                print(percentage_loss)



            if np.array(percentage_loss[:3]).sum() > 100:
                print('percentage_loss 100')


            arr.append([name, 'Negative', -delta_p_p, delta_p_p_sem, delta_p_p_std,  len(name_loss) , big_gain[0], percentage_loss[0], big_gain[1], percentage_loss[1], big_gain[2], percentage_loss[2]])




        elif r_feats[name].mean() - l_feats[name].mean() > 0:
            print('{} has positive correlation '.format(name))    

            name_gain = features_all_fr_only[features_all_fr_only[name]>0]


            name_gain = name_gain.drop(fractions_to_drop, axis = 1)
            

            big_loss = pd.DataFrame(name_gain.mean().sort_values(ascending=True)).T.columns

            delta_p_g = round(features_all.iloc[name_gain.index].target.mean(), 2)
            
            delta_p_g_sem = round(features_all.iloc[name_gain.index].target.sem(), 2)
            
            delta_p_g_std = round(features_all.iloc[name_gain.index].target.std(), 2)

            #percentage_gain=[round(len(name_gain[name_gain[x]<0])/ len(name_gain) *100, 2 ) for x in big_loss[:10]]

            percentage_gain = [round(name_gain[x].sum() / len(name_gain) *100, 2 ) for x in big_loss[:10]]


            # Look at overlaping percentages


            if percentage_gain[1] == percentage_gain[2] == percentage_gain[3]:
                    big_loss=[big_loss[0], (big_loss[1] , big_loss[2], big_loss[3]), big_loss[4] ]
                    percentage_gain = [percentage_gain[0] , percentage_gain[2], percentage_gain[4]]


                    print('1/2/3 loss')
                    print(big_loss)
                    print(percentage_gain)


            elif percentage_gain[0] == percentage_gain[1] == percentage_gain[2]:
                big_loss=[( big_loss[0], big_loss[1] , big_loss[2]) , big_loss[3], big_loss[4]]
                percentage_gain = [percentage_gain[0] , percentage_gain[3], percentage_gain[4]]

                print('0/1/2 loss')
                print(big_loss)
                print(percentage_gain)

            elif int(percentage_gain[0]) == int(percentage_gain[1]):
                big_loss=[(big_loss[0], big_loss[1]), big_loss[2], big_loss[3]]
                percentage_gain = [percentage_gain[0] , percentage_gain[2], percentage_gain[3]]

                print('first double loss')
                print(big_loss)
                print(percentage_gain)

            elif int(percentage_gain[1]) == int(percentage_gain[2]):
                big_loss=[big_loss[0], (big_loss[1] , big_loss[2]) , big_loss[3]]
                percentage_gain = [percentage_gain[0] , percentage_gain[2], percentage_gain[3]]

                print('second double loss')
                print(big_loss)
                print(percentage_gain)




            if np.array(percentage_gain[:3]).sum() < -100:
                print('percentage gain under -100')

            arr.append([name, 'Positive', delta_p_g,  delta_p_g_sem,delta_p_g_std , len(name_gain) , big_loss[0], percentage_gain[0], big_loss[1], percentage_gain[1], big_loss[2], percentage_gain[2]])



    res_neg = pd.DataFrame(arr, columns=[ 'Main fraction', 'Correlation', r'$\overline{\Delta P}$', 'sem','std', 'dof' ,'Opposite fraction 1', '% of opposite 1' ,'Opposite fraction 2', '% of opposite 2','Opposite fraction 3', '% of opposite 3'])

    res_neg = res_neg.sort_values(by='dof', ascending=False)

    res_neg = res_neg.iloc[res_neg[r'$\overline{\Delta P}$'].abs().argsort()[::-1]]


    return res_neg


def generate_mols_for_substructures():

    smarts_substructures =    ['[c][NX3;!$(NC=,#[!#1!#6])]',
     '[a]',
     '[#7;a]',
     '[$(n=[OX1]),$([n+][OX1-])]',
     'c1ccccc1',
     '[!#1!#6;a]',
     '[H]Oc1ccccc1',
     'C=C',
     'C#C',
     'C#[C-]',
     'C=C=C',
     '[#6-]',
     '[$([C;!$(C=[OX1])][NX2-]-[NX2+]#[NX1]),$([C;!$(C=[OX1])][NX2]=[NX2+]=[NX1-]),$([C;!$(C=[OX1])][NX2]=N#[NX1])]',
     '[#1,#6]C(=[NX2])[#7]',
     '[#7][C;!$(C=[!#6])][#7]',
     '[#7X3;!$([#7]C=,#[!#6]);!$([#7]=,#*);!$([#7]S)]',
     '[#6][#7X3H2;!$([#7]C=,#[!#6])]',
     '[#6][#7X3H;!$([#7]C=,#[!#6])][#6]',
     '[#6][#7X3;!$([#7]C=,#[!#6])]([#6])[#6]',
     '[#6][NX4+]',
     '[$([#6][NX2-]-[NX2+]#[NX1]),$([#6][NX2]=[NX2+]=[NX1-]),$([#6][NX2]=N#[NX1])]',
     '[#6;!$([#6]=,#[!#6])]1[#6;!$([#6]=,#[!#6])]N1',
     '[#6][NX2][NX2][#6]',
     '[#6][NX2]=C=[NX2]',
     '[#7]C#[NX1]',
     '[#6]=[NX2]#[NX1]',
     'C=C[#7;!$([#7]C=,#[!#6])]',
     '[#7]C(=N)[#7]',
     '[NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])][NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])]',
     '[#1,#6]C([#1,#6])=[NX2][NX3]',
     '[#1,#6]C([#1,#6])=[NX2;$(N[#1,#6])]',
     'C=[NX2]',
     '[#6][N+]#[C-]',
     'C=C=[NX2]',
     '[#6]C#[NX1]',
     '[$([OX1]=C([#1,#6])[NX2-]-[NX2+]#[NX1]),$([OX1]=C([#1,#6])[NX2]=[NX2+]=[NX1-]),$([OX1]=C([#1,#6])[NX2]=N#[NX1])]',
     '[OX1]=C([#1,#6])C#[NX1]',
     '[OX1]=C([#1,#6])[NX3]C(=[OX1])[#7,#8]',
     '[NX3;!$(NS)]O[#6;!$(C=,#[!#6])]',
     '[$([#6][NX4]([#6])([#6])=[OX1]),$([#6][NX4+]([#6])([#6])[O-])]',
     '[$([#6][NX2]=[NX3+]([O-])[#6]),$([#6][NX2]=[NX3](=[OX1])[#6])]',
     '[NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])]C(=[OX1])O[#6;!$([#6]=,#[!#6])]',
     '[OX1]=[C;$(C[#7]);!$(C[#7]C=[OX1])][#1,#6]',
     '[OX1]=C([#7X3H2])[#1,#6]',
     '[OX1]=[C;$(C[#7X3H][#6,#7,#8]);!$(C[#7]C=[OX1])][#1,#6]',
     '[OX1]=[C;$(C[#7X3]([#6,#7,#8])[#6,#7,#8]);!$(C[#7]C=[OX1])][#1,#6]',
     '[#6]OC#[NX1]',
     'C=C[NX3]C=[OX1]',
     '[#7][C;!$(C=[!#6])][OX2H]',
     '[#7][C;!$(C=[!#6])][OX2][!H]',
     '[OX1]=C([#1,#6])[NX3][OX2H]',
     '[#7X3;!$([#7]C=,#[!#6]);!$([#7]S)][OX2H]',
     '[#1,#6]C(=[NX2])O[#6]',
     '[C;$(C[#1,#6])](=[OX1])[NX3][C;$(C[#1,#6])]=[OX1]',
     '[#6][NX2]=C=[OX1]',
     'O=[#6]-1-[#6]~*~[#6;!$([#6]=,#[!#6])]-[#7]-1',
     '[$([#6]O[NX3](=[OX1])=[OX1]),$([#6]O[NX3+](=[OX1])[O-])]',
     '[#6]O[NX2]=[OX1]',
     '[$([#6][NX3](=[OX1])=[OX1]),$([#6][NX3+](=[OX1])[O-])]',
     '[$([#1,#6]C([#1,#6])=[NX3+]([O-])[#8][#1,#6]),$([#1,#6]C([#1,#6])=[NX3+0](=[OX1])[#8][#1,#6])]',
     '[#1,#6][$(C([#1,#6])=[NX3+]([O-])[#6]),$(C([#1,#6])=[NX3+0](=[OX1])[#6])][#1,#6]',
     '[#6][NX2]=[OX1]',
     '[#1,#6]C([#1,#6])=[NX2][OX2H]',
     '[#1,#6]C([#1,#6])=[NX2]O[#6]',
     '[NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])]C(=[OX1])[NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])]',
     '[NX3;$(N([#1,#6;!$([#6]=,#[!#6])])[#1,#6;!$([#6]=,#[!#6])])]C(=[OX1])O[#6;!$([#6]=,#[!#6])]',
     '[#6;!$([#6]=,#[!#6])]O[C;!$(C=[!#6])]O[#6;!$([#6]=,#[!#6])]',
     '[C;$(C[#1,#6])](=[OX1])O[C;$(C[#1,#6])]=[OX1]',
     '[#1,#6]C(=[OX1])OC(=[OX1])[O,N]',
     '[#1,#6][CH](=[OX1])',
     '[#6;!$([#6]=,#[!#6])][OX2H]',
     'C=,#CC=[OX1]',
     '[#6;!$([#6]=,#[!#6])]OC(=[OX1])O[#6;!$([#6]=,#[!#6])]',
     'C=[OX1]',
     '[OX2H][C;!$(C=[!#6])][OX2H]',
     '[#1,#6]C(=[OX1])[OX2H]',
     '[OX2H][#6;!$([#6]=,#[!#6])]~[#6;!$([#6]=,#[!#6])][OX2H]',
     'C=C[OX2H]',
     '[#1,#6]C(=[OX1])OC=C',
     '[#6;!$([#6]=,#[!#6])]OC=C',
     '[#6;!$([#6]=,#[!#6])]1[#6;!$([#6]=,#[!#6])]O1',
     '[#1,#6]C(=[OX1])O[#6;!$([#6]=,#[!#6])]',
     '[#6;!$([#6]=,#[!#6])]O[#6;!$([#6]=,#[!#6])]',
     '[#6;!$([#6]=,#[!#6])]O[C;!$(C=[!#6])][OX2H]',
     'O[OX2H]',
     'C=C=[OX1]',
     '[#6]C(=[OX1])[#6]',
     '[#1,#6]C(=[OX1])[#1,#6]',
     '[OX2H]-[#6]-1-[#6]~*~[#6;!$([#6]=,#[!#6])]-[#8]-1',
     'O=[#6]-1-[#6]~*~[#6;!$([#6]=,#[!#6])]-[#8]-1',
     '[#6]OO[#6]',
     '[C;$(CO[#6;!$(C=[OX1])])](=[OX1])O[C;$(CO[#6;!$(C=[OX1])])]=[OX1]',
     '[OX1]=C([#1,#6])[BrX1]',
     '[OX1]=C([#1,#6])[ClX1]',
     '[OX1]=C([#1,#6])[FX1]',
     '[OX1]=C([#1,#6])[FX1,ClX1,BrX1,IX1]',
     '[OX1]=C([#1,#6])[IX1]',
     'C=C[BrX1]',
     'C=C[ClX1]',
     'C=C[FX1]',
     'C=C[FX1,ClX1,BrX1,IX1]',
     'C=C[IX1]',
     '[C;!$(C=,#*)][BrX1]',
     '[C;!$(C=,#*)][ClX1]',
     '[C;!$(C=,#*)][FX1]',
     '[C;!$(C=,#*)][FX1,ClX1,BrX1,IX1]',
     '[C;!$(C=,#*)][IX1]',
     'c[BrX1]',
     'c[ClX1]',
     'c[FX1]',
     'c[FX1,ClX1,BrX1,IX1]',
     'c[IX1]',
     '[CX4][$([ClX1,BrX1,IX1]),$(O[SX4](=[OX1])=[OX1]),$(O[SX4+2]([OX1-])[OX1-])]',
     '[#6;!$([#6]#*)][$([ClX1,BrX1,IX1]),$(O[SX4](=[OX1])=[OX1]),$(O[SX4+2]([OX1-])[OX1-])]',
     '[$([#6]O[PX4](=[OX1])(O[#6])O[#6]),$([#6]O[PX4+]([OX1-])(O[#6])O[#6])]',
     '[$([#6][PX4](=[OX1])([#6])O[#6]),$([#6][PX4+]([OX1-])([#6])O[#6])]',
     '[#1,#6][PX3]([#1,#6])[#1,#6]',
     '[$([#6][PX4](=[OX1])([#6])[#6]),$([#6][PX4+]([OX1-])([#6])[#6])]',
     '[$([#6][PX4](=[SX1])([#6])[#6]),$([#6][PX4+]([SX1-])([#6])[#6])]',
     '[$([#6][PX3]([#6])[OX2H]),$([#6][PX4H](=[OX1])[#6]),$([#6][PX4+H]([OX1-])[#6])]',
     '[#6][PX3]([#6])O[#6]',
     '[$([#6][PX3]([#6])[OX2H]),$([#6][PX4H](=[OX1])[#6]),$([#6][PX4+H]([OX1-])[#6])]',
     '[#6]O[PX3](O[#6])O[#6]',
     '[$([#6][PX4](=[OX1])(O[#6])[OX2]),$([#6][PX4+]([OX1-])(O[#6])[OX2])]',
     '[$([#6][PX4](=[OX1])([OX2H])[OX2H]),$([#6][PX4+]([OX1-])([OX2H])[OX2H])]',
     '[$([#6][PX3](O[#6])[OX2]),$([#6][PX4H](=[OX1])O[#6]),$([#6][PX4+H]([OX1-])O[#6])]',
     '[PX4+;!$([PX4+][*-])]',
     '[$([#6][PX3]([OX2H])[OX2]),$([#6][PX4H](=[OX1])[OX2]),$([#6][PX4+H]([OX1-])[OX2])]',
     '[#6][PX4](=C)([#6])[#6]',
     '[#6][SX2][SX2][#6]',
     '[#6][NX2]=C=[SX1]',
     '[$([SX4](=[OX1])(=[OX1])([OX2][#6;!$([#6]=,#[!#6])])[OX2][#6;!$([#6]=,#[!#6])]),$([SX4+2]([OX1-])([OX1-])([OX2][#6;!$([#6]=,#[!#6])])[OX2][#6;!$([#6]=,#[!#6])])]',
     '[#6;!$([#6]=,#[!#6])][SX2][OX2][#6;!$([#6]=,#[!#6])]',
     '[#6;!$([#6]=,#[!#6])][SX2][OX2H]',
     '[#6;!$(C=,#[!#6])][SX2][#6;!$(C=,#[!#6])]',
     '[$([SX3](=[OX1])([#6])[#7X3][!S]),$([SX3+]([OX1-])([#6])[#7X3][!S])]',
     '[#6;!$([#6]=,#[!#6])][SX2][OX2][#6;!$([#6]=,#[!#6])]',
     '[#6;!$([#6]=,#[!#6])][SX2][OX2H]',
     '[$([SX3](=[OX1])([OX2][#6;!$([#6]=,#[!#6])])[OX2][#6;!$([#6]=,#[!#6])]),$([SX3+]([OX1-])([OX2][#6;!$([#6]=,#[!#6])])[OX2][#6;!$([#6]=,#[!#6])])]',
     '[$([SX4](=[OX1])(=[OX1])([#6])[#7X3][!S]),$([SX4+2]([OX1-])([OX1-])([#6])[#7X3][!S])]',
     '[$([SX4](=[OX1])(=[OX1])([#6])[OX2][#6;!$([#6]=,#[!#6])]),$([SX4+2]([OX1-])([OX1-])([#6])[OX2][#6;!$([#6]=,#[!#6])])]',
     '[$([SX4](=[OX1])(=[OX1])([#6])[#6]),$([SX4+2]([OX1-])([OX1-])([#6])[#6])]',
     '[$([SX4](=[OX1])(=[OX1])([#6])[OX2H]),$([SX4+2]([OX1-])([OX1-])([#6])[OX2H])]',
     '[SX3+;!$([SX3+][*-])]',
     '[$([SX3](=[OX1])([#6])[#6]),$([SX3+]([OX1-])([#6])[#6])]',
     'C=[SX1]',
     '[#1,#6]C(=[SX1])[#7]',
     '[#6][SX2]C#[NX1]',
     '[#1,#6]C(=[OX1])[SX2][#6;!$([#6]=,#[!#6])]',
     '[#6;!$(C=,#[!#6])][SX2H]',
     '[#7]C(=[SX1])[#7]',
     '[#6;!$([#6]=,#[!#6])]OC(=[SX1])[SX2][#6;!$([#6]=,#[!#6])]',
      's1cccc1', 
      '[#1,#6]C(=[OX1])[OX2H]']
    
    name_substructure = ['aniline',
     'arene',
     'azaarene',
     'azaarene oxide',
     'benzene ring',
     'heteroarene',
     'phenol',
     'alkene',
     'alkyne',
     'alkynylide',
     'allene',
     'carbanion',
     'alkyl azide',
     'amidine',
     'aminal',
     'amine',
     'primary amine',
     'secondary amine',
     'tertiary amine',
     'ammonium',
     'azide',
     'aziridine',
     'azo',
     'carbodiimide',
     'cyanamide',
     'diazo',
     'enamine',
     'guanidine',
     'hydrazine',
     'hydrazone',
     'imine',
     'iminyl',
     'isonitrile',
     'ketenimine',
     'nitrile',
     'acyl azide',
     'acyl nitrile',
     'N-acylcarbamate or urea (mixed imide)',
     'alkoxylamine',
     'amine oxide, tertiary',
     'azoxy',
     'carbamate',
     'carboxamide',
     'primary carboxamide',
     'secondary carboxamide',
     'tertiary carboxamide',
     'cyanate',
     'enamide',
     'hemiaminal',
     'hemiaminal ether',
     'hydroxamic acid',
     'hydroxylamine',
     'imidate ester',
     'imide',
     'isocyanate',
     'lactam',
     'nitrate ester',
     'nitrite ester',
     'nitro',
     'nitronate ester',
     'nitrone',
     'nitroso',
     'oxime',
     'oxime ether',
     'urea',
     'urethane',
     'acetal',
     'acyl anhydride',
     'O-acylcarbonate or carbamate (mixed anhydride)',
     'aldehyde',
     'alkanol',
     'α,β-unsaturated carbonyl',
     'carbonate ester',
     'carbonyl',
     'carbonyl hydrate (1,1-diol)',
     'carboxylic acid',
     '1,2-diol',
     'enol',
     'enol ester',
     'enol ether',
     'epoxide',
     'ester (carboxylate ester)',
     'ether',
     'hemiacetal',
     'hydroperoxide',
     'ketene',
     'ketone',
     'ketone or aldehyde',
     'lactol',
     'lactone',
     'peroxide',
     'pyrocarbonate diester',
     'acyl bromide',
     'acyl chloride',
     'acyl fluoride',
     'acyl halide',
     'acyl iodide',
     'alkenyl bromide',
     'alkenyl chloride',
     'alkenyl fluoride',
     'alkenyl halide',
     'alkenyl iodide',
     'alkyl bromide',
     'alkyl chloride',
     'alkyl fluoride',
     'alkyl halide',
     'alkyl iodide',
     'aryl bromide',
     'aryl chloride',
     'aryl fluoride',
     'aryl halide',
     'aryl iodide',
     'C(sp3)-leaving group',
     'leaving group',
     'phosphate triester',
     'phosphinate ester',
     'phosphine',
     'phosphine oxide',
     'phosphine sulfide',
     'phosphinic acid',
     'phosphinite ester',
     'phosphinous acid',
     'phosphite triester',
     'phosphonate monoester or diester',
     'phosphonic acid or monoester',
     'phosphonite monoester or diester',
     'phosphonium',
     'phosphonous acid or monoester',
     'phosphorane',
     'disulfide',
     'isothiocyanate',
     'sulfate diester',
     'sulfenate ester',
     'sulfenic acid',
     'sulfide',
     'sulfinamide',
     'sulfinate ester',
     'sulfinic acid',
     'sulfite diester',
     'sulfonamide',
     'sulfonate ester',
     'sulfone',
     'sulfonic acid',
     'sulfonium',
     'sulfoxide',
     'thiocarbonyl',
     'thiocarboxamide',
     'thiocyanate',
     'thioester',
     'thiol',
     'thiourea',
     'xanthate ester',
      'thiophene',
    'carboxylic acid']

    #fetch all substructure definitions and calculate mosl for them
    print('Generating molecular objects from pre-defined substructures')
    mol_substructures=[]
    for substructure in smarts_substructures:
        mol_substructures.append(Chem.MolFromSmarts(substructure))

    return mol_substructures,  name_substructure


def find_sig_feats_mk2(l_feats, r_feats, p_val):
    
    df_sig_feats=pd.DataFrame(columns=l_feats.columns)
    
    #df_feats_fr_only=df_feats.iloc[:,114:199]

    for name in df_sig_feats.columns[:-3]:
        t, p = stats.ttest_rel(r_feats[name],l_feats[name])
        if p < p_val:
            df_sig_feats[name]=pd.Series([p, t])

    df_sig_feats = df_sig_feats.dropna(axis=1)
    
    print('Found significant fractions: ', len(df_sig_feats.columns))

    return df_sig_feats



def results_arr_dev(features_all, fr_sig_descriptors, r_feats, l_feats , fractions_to_drop  ):

    arr=[]

    features_all_fr_only = features_all.iloc[:,:-2]

    for name in fr_sig_descriptors:
        if r_feats[name].mean() - l_feats[name].mean() == 0:
            print('{} has neutral correlation '.format(name))


        elif r_feats[name].mean() - l_feats[name].mean() < 0:

            print('{} has negative correlation '.format(name))
            name_loss= features_all_fr_only[features_all_fr_only[name]<0] # find all negative occurances of this named fraction

            name_loss = name_loss.drop(fractions_to_drop, axis = 1)

            # in all fractions that name is lost who are biggest gains:
            big_gain = pd.DataFrame(name_loss.mean().sort_values(ascending=False)).T.columns
            
          
            delta_p_p = round(features_all.iloc[name_loss.index].target.mean(), 2)
            
            percentage_loss = [round(name_loss[x].sum() / len(name_loss) *100, 2 ) for x in big_gain[:10]]



            if np.array(percentage_loss[:3]).sum() > 100:
                print('percentage_loss 100')


            arr.append([name, 'Negative', -delta_p_p,  len(name_loss) , big_gain[0], percentage_loss[0], big_gain[1], percentage_loss[1], big_gain[2], percentage_loss[2], big_gain[3], percentage_loss[3], big_gain[4], percentage_loss[4], big_gain[5], percentage_loss[5], big_gain[6], percentage_loss[6]])




        elif r_feats[name].mean() - l_feats[name].mean() > 0:
            print('{} has positive correlation '.format(name))

            name_gain = features_all_fr_only[features_all_fr_only[name]>0]


            name_gain = name_gain.drop(fractions_to_drop, axis = 1)


            big_loss = pd.DataFrame(name_gain.mean().sort_values(ascending=True)).T.columns

            delta_p_g = round(features_all.iloc[name_gain.index].target.mean(), 2)

            percentage_gain = [round(name_gain[x].sum() / len(name_gain) *100, 2 ) for x in big_loss[:10]]


            if np.array(percentage_gain[:3]).sum() < -100:
                print('percentage gain under -100')

            arr.append([name, 'Positive', delta_p_g,  len(name_gain) , big_loss[0], percentage_gain[0], big_loss[1], percentage_gain[1], big_loss[2], percentage_gain[2], big_loss[3], percentage_gain[3], big_loss[4], percentage_gain[4], big_loss[5], percentage_gain[5], big_loss[6], percentage_gain[6]])



    res_neg = pd.DataFrame(arr, columns=[ 'Main fraction', 'Correlation', r'$\overline{\Delta P}$', 'dof' ,'Opposite fraction 1', '% of opposite 1' ,'Opposite fraction 2', '% of opposite 2','Opposite fraction 3', '% of opposite 3','Opposite fraction 4', '% of opposite 4','Opposite fraction 5', '% of opposite 5','Opposite fraction 6', '% of opposite 6','Opposite fraction 7', '% of opposite 7'])

    res_neg = res_neg.sort_values(by='dof', ascending=False)

    res_neg = res_neg.iloc[res_neg[r'$\overline{\Delta P}$'].abs().argsort()[::-1]]


    return res_neg


def calculate_fractions_mk3(df):

    # Generate substructure mols and names

    mol_substructures, name_substructure = generate_mols_for_substructures()

    name_substructure = name_substructure + ['smirks', 'target']

    # Comapre left hand side

    #     frame_left=pd.DataFrame(columns=name_substructure)

    frame_left=[]

    print('Calcualting LHS matches')

    for num, target in tqdm(enumerate(df.LHS.values)):
        #grab structure
        frame_temp=pd.DataFrame(0, index=range(1), columns=name_substructure)
        #turn it into mol 
        try:
            mol_target=Chem.MolFromSmarts(target)
            mol_target.UpdatePropertyCache()
            mol_target = Chem.AddHs(mol_target)
        except TypeError:
            print('Error: ', num, target)


        for index, sub in enumerate(mol_substructures):
            if mol_target.HasSubstructMatch(sub):
                #print('yeah', smarts_names[index])
                frame_temp[name_substructure[index]] = [1]
                frame_temp['smirks'] = df.transition.values[num]
                frame_temp['target'] = df.avg_pred_diff.values[num]

        frame_left.append(frame_temp.values)



    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side

    #     frame_right=pd.DataFrame(columns=name_substructure)
    frame_right=[]

    print('Calcualting RHS matches')

    for num, target in tqdm(enumerate(df.RHS.values)):

        frame_temp=pd.DataFrame(0, index=range(1), columns=name_substructure)
        try:
            mol_target=Chem.MolFromSmarts(target)
            mol_target.UpdatePropertyCache()
            mol_target = Chem.AddHs(mol_target)
        except TypeError:
            print('Error Right', num, target)
        for index, sub in enumerate(mol_substructures):
            if mol_target.HasSubstructMatch(sub):
                #print('yeah', smarts_names[index])
                frame_temp[name_substructure[index]] = [1]
                frame_temp['smirks'] = df.transition.values[num]
                frame_temp['target'] = df.avg_pred_diff.values[num]

        frame_right.append(frame_temp.values)


    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-2] - frame_left_df.iloc[:,:-2] 

    diff['smirks'] = frame_right_df['smirks']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)


# Generate substructure mols and names


def calculate_fractions_mk4(df):

    mol_substructures, name_substructure = generate_mols_for_substructures()

    name_substructure = name_substructure + ['smirks', 'target']

    # Comapre left hand side

    #     frame_left=pd.DataFrame(columns=name_substructure)

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS+RHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in tqdm(range(len(df))):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)
        frame_temp_right=pd.DataFrame(0, index=range(1), columns=name_substructure)

        #turn it into mol 
        try:
            mol_target_left=Chem.MolFromSmarts(df.LHS.values[index])
            mol_target_left.UpdatePropertyCache()
            mol_target_left = Chem.AddHs(mol_target_left)
        except TypeError:
            print('Error: ', index, target)

        try:
            mol_target_right=Chem.MolFromSmarts(df.RHS.values[index])
            mol_target_right.UpdatePropertyCache()
            mol_target_right = Chem.AddHs(mol_target_right)
        except TypeError:
            print('Error: ', index, target)    

        if type(mol_target_right) != Chem.rdchem.Mol or type(mol_target_left) != Chem.rdchem.Mol:

            print('failed to MolObject: ', index)


        frame_temp_left['smirks'] = df.smirks.values[index]
        frame_temp_left['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_left.HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_temp_right['smirks'] = df.smirks.values[index]
        frame_temp_right['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_right.HasSubstructMatch(sub):
                frame_temp_right[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)
        frame_right.append(frame_temp_right.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side
    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-2] - frame_left_df.iloc[:,:-2] 

    diff['smirks'] = frame_right_df['smirks']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)

def calculate_fractions_single_mk4(df, col):

    mol_substructures, name_substructure = generate_mols_for_substructures()

    name_substructure = name_substructure + ['Smiles']

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in tqdm(range(len(df))):  
        
        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)

        #turn it into mol 
        try:
            mol_target_left=Chem.MolFromSmarts(df[col].values[index])
            mol_target_left.UpdatePropertyCache()
            mol_target_left = Chem.AddHs(mol_target_left)
        except TypeError:
            print('Error: ', index, target)  

        if type(mol_target_left) != Chem.rdchem.Mol:

            print('failed to MolObject: ', index)

        frame_temp_left['Smiles'] = df[col].values[index]

        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_left.HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)

    return frame_left_df.reset_index(drop=True)

def calculate_fractions_single_mk7(df): # must contain 'mol' colum

    mol_substructures, name_substructure = generate_mols_for_substructures()

#     name_substructure = name_substructure + ['Smiles']

    frame=[]

    for index in range(len(df)):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)

    #     frame_temp_left['Smiles'] = df[col].values[index]

        for sub_nr, sub in enumerate(mol_substructures):
            if df['mol'].iloc[index].HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame.append(frame_temp_left.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame), columns = name_substructure)

    return frame_left_df.reset_index(drop=True)

def calculate_fractions_mk7(df):
    print('must contain mol_a and mol_b')
    mol_substructures, name_substructure = generate_mols_for_substructures()

    name_substructure = name_substructure + ['smirks', 'target']

    frame_left=[]
    frame_right=[]

    for index in range(len(df)):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)
        frame_temp_right=pd.DataFrame(0, index=range(1), columns=name_substructure)

        frame_temp_left['smirks'] = df.smirks.values[index]
        frame_temp_left['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if df['mol_a'].iloc[index].HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_temp_right['smirks'] = df.smirks.values[index]
        frame_temp_right['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if df['mol_b'].iloc[index].HasSubstructMatch(sub):
                frame_temp_right[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)
        frame_right.append(frame_temp_right.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side
    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-2] - frame_left_df.iloc[:,:-2] 

    diff['smirks'] = frame_right_df['smirks']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)


def calculate_fractions_mk7_new_smarts(df):
    
    mol_substructures, name_substructure = new_smarts()

    name_substructure = name_substructure + ['smirks', 'measurement' ,'target']

    # Comapre left hand side

    #     frame_left=pd.DataFrame(columns=name_substructure)

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS+RHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in range(len(df)):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)
        frame_temp_right=pd.DataFrame(0, index=range(1), columns=name_substructure)

        frame_temp_left['smirks'] = df.smirks.values[index]
        frame_temp_left['target'] = df.measurement_delta.values[index]    
        frame_temp_left['measurement'] = df.measurement_A.values[index]
        
        
        for sub_nr, sub in enumerate(mol_substructures):
            if df['mol_a'].iloc[index].HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]
            
#             if mol_target_left.HasSubstructMatch(sub):
#                 frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_temp_right['smirks'] = df.smirks.values[index]
        frame_temp_right['target'] = df.measurement_delta.values[index]    
        frame_temp_right['measurement'] = df.measurement_B.values[index]
        
        for sub_nr, sub in enumerate(mol_substructures):
            if df['mol_b'].iloc[index].HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)
        frame_right.append(frame_temp_right.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side
    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-3] - frame_left_df.iloc[:,:-3] 

    diff['smirks'] = frame_right_df['smirks']
    diff['measurement_A'] = frame_left_df['measurement']
    diff['measurement_B'] = frame_right_df['measurement']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)


def new_smarts():
    print(os.getcwd())
    func_groups=pd.read_csv('ml_mmpa/fg_smarts_2.csv')
    
        #fetch all substructure definitions and calculate mosl for them
    print('Generating molecular objects from pre-defined substructures')
    mol_substructures=[]
    for substructure in func_groups.SMARTS:
        mol_substructures.append(Chem.MolFromSmarts(substructure))

    return mol_substructures,  func_groups.name.to_list()

def new_smarts_2():
    print(os.getcwd())
    func_groups=pd.read_csv('ml_mmpa/master_smarts_pattern.csv')
    
        #fetch all substructure definitions and calculate mosl for them
    print('Generating molecular objects from pre-defined substructures')
    mol_substructures=[]
    for substructure in func_groups.SMARTS:
        mol_substructures.append(Chem.MolFromSmarts(substructure))

    return mol_substructures,  func_groups.Description.to_list()

def calculate_fractions_mk5(df):
    
    mol_substructures, name_substructure = new_smarts()

    name_substructure = name_substructure + ['smirks', 'measurement' ,'target']

    # Comapre left hand side

    #     frame_left=pd.DataFrame(columns=name_substructure)

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS+RHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in tqdm(range(len(df))):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)
        frame_temp_right=pd.DataFrame(0, index=range(1), columns=name_substructure)

        #turn it into mol 
        try:
            mol_target_left=Chem.MolFromSmarts(df.LHS.values[index])
            mol_target_left.UpdatePropertyCache()
            mol_target_left = Chem.AddHs(mol_target_left)
            Chem.SanitizeMol(mol_target_left)
        except Chem.KekulizeException:
            print('Error: ', index, target)

        try:
            mol_target_right=Chem.MolFromSmarts(df.RHS.values[index])
            mol_target_right.UpdatePropertyCache()
            mol_target_right = Chem.AddHs(mol_target_right)
            Chem.SanitizeMol(mol_target_right)
        except Chem.KekulizeException:
            print('Error: ', index, target)    

        if type(mol_target_right) != Chem.rdchem.Mol or type(mol_target_left) != Chem.rdchem.Mol:

            print('failed to MolObject: ', index)


        frame_temp_left['smirks'] = df.smirks.values[index]
        frame_temp_left['target'] = df.measurement_delta.values[index]    
        frame_temp_left['measurement'] = df.measurement_A.values[index]
        
        
        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_left.HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_temp_right['smirks'] = df.smirks.values[index]
        frame_temp_right['target'] = df.measurement_delta.values[index]    
        frame_temp_right['measurement'] = df.measurement_B.values[index]
        
        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_right.HasSubstructMatch(sub):
                frame_temp_right[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)
        frame_right.append(frame_temp_right.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side
    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-3] - frame_left_df.iloc[:,:-3] 

    diff['smirks'] = frame_right_df['smirks']
    diff['measurement_A'] = frame_left_df['measurement']
    diff['measurement_B'] = frame_right_df['measurement']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)


def calculate_fractions_mk6(df):
    
    mol_substructures, name_substructure = new_smarts_2()

    name_substructure = name_substructure + ['smirks', 'target']

    # Comapre left hand side

    #     frame_left=pd.DataFrame(columns=name_substructure)

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS+RHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in tqdm(range(len(df))):  

        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)
        frame_temp_right=pd.DataFrame(0, index=range(1), columns=name_substructure)

        #turn it into mol 
        try:
            mol_target_left=Chem.MolFromSmarts(df.LHS.values[index])
            mol_target_left.UpdatePropertyCache()
            mol_target_left = Chem.AddHs(mol_target_left)
            Chem.SanitizeMol(mol_target_left)
        except Chem.KekulizeException:
            print('Error: ', index, target)

        try:
            mol_target_right=Chem.MolFromSmarts(df.RHS.values[index])
            mol_target_right.UpdatePropertyCache()
            mol_target_right = Chem.AddHs(mol_target_right)
            Chem.SanitizeMol(mol_target_right)
        except Chem.KekulizeException:
            print('Error: ', index, target)    

        if type(mol_target_right) != Chem.rdchem.Mol or type(mol_target_left) != Chem.rdchem.Mol:

            print('failed to MolObject: ', index)


        frame_temp_left['smirks'] = df.smirks.values[index]
        frame_temp_left['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_left.HasSubstructMatch(sub):
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_temp_right['smirks'] = df.smirks.values[index]
        frame_temp_right['target'] = df.measurement_delta.values[index]    

        for sub_nr, sub in enumerate(mol_substructures):
            if mol_target_right.HasSubstructMatch(sub):
                frame_temp_right[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)
        frame_right.append(frame_temp_right.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)
    # compare right hand side
    frame_right_df = pd.DataFrame(np.concatenate(frame_right), columns = name_substructure)

    diff = frame_right_df.iloc[:,:-2] - frame_left_df.iloc[:,:-2] 

    diff['smirks'] = frame_right_df['smirks']
    diff['target'] = frame_right_df['target']

    return diff.reset_index(drop=True), frame_left_df.reset_index(drop=True), frame_right_df.reset_index(drop=True)

def calculate_fractions_single_new_smarts_mk7(df, mol_col):

    mol_substructures, name_substructure = new_smarts()

    name_substructure = name_substructure + ['Smiles']

    frame_left=[]
    frame_right=[]

    print('Calcualting LHS matches')

    #for index in enumerate(df.LHS.values)):

    for index in tqdm(range(len(df))):  
        
        #grab structure
        frame_temp_left=pd.DataFrame(0, index=range(1), columns=name_substructure)

        for sub_nr, sub in enumerate(mol_substructures):
            if df[mol_col].iloc[index].HasSubstructMatch(sub):                
                frame_temp_left[name_substructure[sub_nr]] = [1]

        frame_left.append(frame_temp_left.values)

    frame_left_df = pd.DataFrame(np.concatenate(frame_left), columns = name_substructure)

    return frame_left_df.reset_index(drop=True)


def tsne_no_plot(df, perp):
    
    sample=df.values
    tanimoto_sim_mat_lower_triangle=GetTanimotoSimMat(sample) # similartity matrix 
    n_mol = len(sample)
    similarity_matrix = np.ones([n_mol,n_mol])
    i_lower= np.tril_indices(n=n_mol,m=n_mol,k=-1)
    i_upper= np.triu_indices(n=n_mol,m=n_mol,k=1)
    similarity_matrix[i_lower] = tanimoto_sim_mat_lower_triangle
    similarity_matrix[i_upper] = similarity_matrix.T[i_upper] 
    distance_matrix = np.subtract(1,similarity_matrix) 

    TSNE_sim = TSNE(verbose=1, n_components=2, init='pca', method='barnes_hut', perplexity=perp).fit_transform(distance_matrix) 

    tsne_result = pd.DataFrame(data = TSNE_sim , columns=["TC1","TC2"]) 
    return tsne_result

def calc_feats(df):
    table=pd.DataFrame()
#     df=df.dropna() 
    
    for i,mol in enumerate(df):
#         Chem.SanitizeMol(mol)
#         table.loc[i,'SMILES']=Chem.MolToSmiles(mol)
#         table.loc[i,'Mol']=mol
        table.loc[i,'MolWt']=Descriptors.MolWt(mol)
        table.loc[i,'LogP']=Descriptors.MolLogP(mol)
        table.loc[i,'NumHAcceptors']=Descriptors.NumHAcceptors(mol)
        table.loc[i,'NumHDonors']=Descriptors.NumHDonors(mol)
        table.loc[i,'NumHeteroatoms']=Descriptors.NumHeteroatoms(mol)
        table.loc[i,'NumRotatableBonds']=Descriptors.NumRotatableBonds(mol)
        table.loc[i,'NumHeavyAtoms']=Descriptors.HeavyAtomCount (mol)
        table.loc[i,'NumAliphaticCarbocycles']=Descriptors.NumAliphaticCarbocycles(mol)
        table.loc[i,'NumAliphaticHeterocycles']=Descriptors.NumAliphaticHeterocycles(mol)
        table.loc[i,'NumAliphaticRings']=Descriptors.NumAliphaticRings(mol)
        table.loc[i,'NumAromaticCarbocycles']=Descriptors.NumAromaticCarbocycles(mol)
        table.loc[i,'NumAromaticHeterocycles']=Descriptors.NumAromaticHeterocycles(mol)
        table.loc[i,'NumAromaticRings']=Descriptors.NumAromaticRings(mol)
        table.loc[i,'RingCount']=Descriptors.RingCount(mol)
        table.loc[i,'FractionCSP3']=Descriptors.FractionCSP3(mol)
        table.loc[i,'TPSA']=Descriptors.TPSA(mol)
    
    return table