#!/usr/bin/env python
# coding: utf-8

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time

#############  choose which molecule to be processed #############
molecule = "SO3"
##################################################################
# linux command to run this code:
# nohup python3 nohup_calculations.py > ./log/nohup_calculations.log 2>&1 &


date = time.strftime("%d/%m/%Y")
date = date.replace('/','-')
start_time = time.time()

## load infomation from def file 
iso_info = pd.read_pickle("molecule_first_iso_final.pickle")

# ignore some molecules that cannot be processed at present
all_molecules = set(iso_info["molecule"])
diff_merged = pd.read_csv("diff_merged_utf8.csv")
wrong_molecules = set(diff_merged["molecule"])
add_wrong_modules = {'PH','H2','LiH',"CO2","H3O_p","LiH_p","CH","MgH","CH","trans-P2H2"} #known issues about the files or wrong headers or the ground state energy is not 0
correct_molecules = all_molecules - wrong_molecules # ignore the molecules with wrong headers
correct_molecules = correct_molecules -add_wrong_modules 
# ignore the molecules whose recommended linelist is MoLLIST since they are not complete which may lead to some very strange vibrational lifetimes.
MoLLIST = set(iso_info.loc[iso_info["linelist"]=='MoLLIST','molecule']) 
correct_molecules = correct_molecules - MoLLIST
correct_molecules = correct_molecules | {"CH","NH3","CO2","H3O_p","SiH","H3_p","SiO2","VO","SiH2","H2O","AlO","H2S"}

print("The following molecules can be processed:")
print(" ")
print(list(correct_molecules))
print(" ")

if molecule not in correct_molecules:
    raise IndexError("The header of this molecule in the def file is not consistent with the actual states file!")

iso_slug = iso_info.loc[iso_info["molecule"]==molecule,"iso_slug"]
isotopologue = iso_info.loc[iso_info["molecule"]==molecule,"linelist"]
linelist = iso_info.loc[iso_info["molecule"]==molecule,"linelist"].values[0]
trunc_E = iso_info.loc[iso_info["molecule"]==molecule,"truncate_E"]
trans_line_num = iso_info.loc[iso_info["molecule"]==molecule,"trans_lines_num_total"]
trans_file_num = int(iso_info.loc[iso_info["molecule"]==molecule,"trans_files_num"])
iso_info.loc[iso_info["molecule"]==molecule ,"headers"].values[0]

    
# create files to store the results

if os.path.exists('original_result'):
    pass
else:
    os.makedirs('original_result', exist_ok=True)

#if os.path.exists('v1_result'):
#    pass
#else:
#    os.makedirs('v1_result', exist_ok=True)


if os.path.exists('v3_result'):
    pass
else:
    os.makedirs('v3_result', exist_ok=True)

if os.path.exists('decay_result'):
    pass
else:
    os.makedirs('decay_result', exist_ok=True)
    
#if os.path.exists('decay_result/'+ molecule+'/v1'):
#    pass
#else:
#    os.makedirs('decay_result/'+ molecule+'/v1', exist_ok=True)
    
if os.path.exists('decay_result/'+ molecule+'/v3'):
    pass
else:
    os.makedirs('decay_result/'+ molecule+'/v3', exist_ok=True)
    
if os.path.exists('compare_result'):
    pass
else:
    os.makedirs('compare_result', exist_ok=True)
    
if os.path.exists('plot'):
    pass
else:
    os.makedirs('plot', exist_ok=True)

if os.path.exists('compute_info'):
    pass
else:
    os.makedirs('compute_info', exist_ok=True)

path_mol_iso_list = list(molecule+'/'+iso_slug+'/'+isotopologue)
path_mol_iso = path_mol_iso_list[0]
read_path = '../../exomol/exomol3_data/'


## read states file
# manually adjust headers due to incorrect def file
print("Reading the .states file ...")
states_col_name = iso_info.loc[iso_info["molecule"]==molecule ,"headers"].values[0]

# manually correct headers for some molecules whose headers from the def files are not right
if molecule =="NH3":
    states_col_name = ['i', 'E_i', 'g_i', 'J_i','unc', '+/-', 'Gamma', 'N(Block)', 'n1', 'n2', 'n3', 'n4', 'l3', 'l4', 'tau(inv)', 'J', 'K', 'Gamma(rot)', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'Gamma(vib)','Eclac']
if molecule =="H3O_p":
    states_col_name = ['i','E_i','gtot','J_i','unc','Gamma','n1','n2','n3','l3','n4','l4','Gamma_vib','K','Gamma_rot','Ci','v1','v2','v3','v4','v5','v6','Elac']
if molecule == "CO2":
    states_col_name = ['i','E_i','gtot','J_i','unc','Gamma','e/f','n1','n2lin','l2','n3','Ci','m1','m2','m3','m4','r','v1','v2','v3']
if molecule == "SiH":
    states_col_name = ['i','E_i','g_i','J_i','tau_i','g-factor','Parity','e/f','State','v','Lambda','Sigma','Omega']
if molecule == "SiH2":
    states_col_name = ['i','E_i','g_i','J_i','Gamma_tot','v1','v2','v3','Gamma_vib','Ka','Kc','Gamma_rot','C','n1','n2','n3']
if molecule == "H3_p":
    states_col_name = ['i','E_i','g','J_i','tau_i','p','Sym','v1','v2','l2','G','U','K']
if molecule == "SiO2":
    states_col_name = ['i','E_i','gtot','J_i','unc','Gamma_tot','e/f','v1','v2lin','L','v3','Ci','n1','n2','n3','Gamma_vib','K','Gamma_rot']
if molecule == "CH":
    states_col_name =  ['i', 'E_i', 'g_i', 'J_i', 'v','Omega', 'e/f', 'State'] # not sure
if molecule == "H2CO":
    states_col_name =  ['i', 'E_i', 'g_i', 'J_i','unc','tau_i', 'G', 'v1', 'v2', 'v3', 'v4','v5', 'v6', 'Gv', 'Ja', 'K', 'Pr', 'Gr', 'N(B1)', 'C2', 'n1', 'n2','n3', 'n4', 'n5', 'n6']
if molecule == "VO":
    states_col_name = ['i','E_i','g_i','J_i','tau_i','+/-','e/f','State','v','Lambda','Sigma','Omega']
if molecule == "AlO":
    states_col_name = ['i','E_i','g_i','J_i','unknown','tau_i','+/-','e/f','State','v','Lambda','Sigma','Omega','EH']
if molecule == "H2S":
    states_col_name =['i','E_i','g_i','J_i','tau_i','Gamma','Ka','Kc','v1','v2','v3','e/c']
if molecule == "H2O":
    states_col_name = ['i', 'E_i', 'g_i', 'J_i', 'N', 'Ka', 'Kc', 'v1', 'v2', 'v3','S', 'Gamma_rve'] 
if molecule == "H2O2":
    states_col_name = ['i', 'E_i', 'g_i', 'J_i', 'tau_i', 'G', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'tau', 'Gv', 'K', 'Pr', 'Gr', 'N(Bl', 'C2', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
if molecule == "NaH":
    states_col_name = ['i', 'E_i', 'g_i', 'J_i', 'tau_i','State','v']


s_df = dict()
states_df = pd.DataFrame()
states_filenames = glob.glob(read_path + path_mol_iso + '/' + path_mol_iso.split('/')[1]+ '__' + path_mol_iso.split('/')[2] + '.states.bz2')
print('states_filenames:',states_filenames)
if len(states_filenames) >0 :
    states_filename = states_filenames[0]
    s_df[states_filename] =pd.read_csv(states_filename, compression='bz2', sep='\s+',
                                            header=None, names=states_col_name,
                                            chunksize=100_000_000, iterator=True,
                                            low_memory=False)
else:
    raise IndexError("Cannot find .states files on your device!")

for chunk in s_df[states_filename]:
    states_df = states_df.append(chunk)

print('states_df_row:',states_df.iloc[0])

states_df_complete = states_df.copy()
# judge which labels to keep
# drop all the records with negative states
colnames = states_df.columns.values.tolist()
print('colnames:',colnames)

if 'V' in colnames:
    states_df = states_df.rename(columns={'V':'v'})

if len({"State",'v','Lambda','Sigma','Omega'} & set(colnames)) !=0:
    labels = list({"State",'v','Lambda','Sigma','Omega'} & set(colnames))
    states = ["State",'v','Lambda','Sigma','Omega']
    labels = sorted(labels,key = states.index)
    states_df.drop(states_df[states_df.v<0].index,inplace= True)

elif ('v1' in colnames) and ('n1' not in colnames):
    labels = []
    if 'l' in colnames:
        labels.insert(0,'l')
    p_v1 = list(colnames).index('v1')
    for i in range(p_v1,len(colnames)):
        if colnames[i].startswith('v') and colnames[i] != 'vibSym':
            labels.append(colnames[i])
            states_df.dropna(subset = [colnames[i]],inplace= True)
            states_df.drop(states_df[states_df[colnames[i]]=="*"].index,inplace= True)
            states_df[colnames[i]] = [np.float(x)for x in states_df[colnames[i]]]
            states_df.drop(states_df[states_df[colnames[i]]<0].index,inplace= True)    
    
elif ('n1' in colnames) and ('v1' not in colnames):
    labels = []
    if 'l' in colnames:
        labels.insert(0,'l')
    p_n1 = list(colnames).index('n1')
    for i in range(p_n1,len(colnames)):
        if colnames[i].startswith('n'):
            labels.append(colnames[i])
            states_df.drop(states_df[states_df[colnames[i]]<0].index,inplace= True)
    
elif ('v1' in colnames) and ('n1' in colnames):
    labels = []
    if 'l' in colnames:
        labels.insert(0,'l')
    p_v1 = list(colnames).index('v1')
    p_n1 = list(colnames).index('n1')
    p_min = np.min([p_v1,p_n1]) # choose the one which appears first, n1 or v1
    p_max = np.max([p_v1,p_n1]) # choose the one which appears second, n1 or v1

    if p_min == p_v1:
        start = 'v'
    else:
        start = 'n'
    for i in range(p_min,p_max):
        if (colnames[i].startswith(start)) or (colnames[i].startswith('l')) or  (colnames[i].startswith('L')) or  (colnames[i].startswith('M')) or  (colnames[i].startswith('m')): #and (colnames[i] != 'vibSym'):
            labels.append(colnames[i])
            states_df.drop(states_df[states_df[colnames[i]]<0].index,inplace= True)
else:
    labels = []
    
print(labels)
    


# Calculate the J that maximize the Boltzmann distribution P(J) - using alternative method
# P(J) = (2*J+1)*exp(-(rotational_energy)/(kT))
# where rotational_energy = E(v,J) - E(v,J=0.5) for each state v
# select the J which maximize P(J)

# Drop the records with v = NaN
states_df = states_df.dropna(subset=labels)#.groupby(labels).agg({'E_i':'mean'})
# Only use records with v = 0 so usually  E(v,J= Jmin) should be 0 (ground state). Will report if that energy is not 0.
ground_E = float(states_df.loc[states_df['i'] ==1,"E_i"])
if ground_E != 0:
    raise ValueError("The energy at the ground state is",ground_E,",rather than 0.")
    
v0 = list((states_df.loc[states_df["i"] ==1,labels]).values[0])
print('v0 = ',v0)
states_df_v0_indx = list(states_df[labels][states_df[labels] == v0].dropna(subset = labels).index)
states_df_v0 = states_df.loc[states_df.index.isin(states_df_v0_indx)]

k = 1.438775 
T = 500 # Tempertature = 500K

def rot_energy(dataset_v):
    Jmin_loc = np.min(dataset_v["J_i"])
    E_v_J0 =  np.min(dataset_v.loc[(dataset_v["J_i"]==Jmin_loc),"E_i"].values) # should be 0
    return [E_v_J0]*len(dataset_v)


energy = states_df_v0.groupby(labels).apply(rot_energy)

E_v_J0_list = []
for i in range(len(energy)):
    E_v_J0_list += energy.iloc[i]

states_df_v0 = states_df_v0.sort_values(by= labels,ascending = True)
states_df_v0["E_v_J0"] = E_v_J0_list

#v = states_df["v"]
J = states_df_v0["J_i"]
E_v_J0 =  E_v_J0_list
rot_E = states_df_v0["E_i"] - E_v_J0
P_J = (2*J+1)*np.exp(-rot_E/(k*T))
log_P_J = np.log(2*J+1)-rot_E/(k*T)
states_df_v0["rot_E"] = rot_E
states_df_v0["P_J"] = P_J
states_df_v0["log_P_J"] = log_P_J

log_max = states_df_v0["log_P_J"].max()
states_df_v0.loc[states_df_v0["log_P_J"]==log_max]
J_argmax = list(states_df_v0.loc[states_df_v0["log_P_J"]==log_max,'J_i'])[0]
print("J that maximizes P(J) is",J_argmax,"at T =",T,"K")

## read .trans files
print("Reading the .trans file ...")
trans_col_name = ['i', 'f', 'A_if']
t_df = dict()
#trans_filenames = sorted(glob.glob(read_path + path_mol_iso+"/"+"*"+linelist+".trans.bz2"))
trans_filenames = sorted(glob.glob(read_path + path_mol_iso+"/"+"*.trans.bz2"))
if len(trans_filenames) != trans_file_num:
    raise IndexError("There are other weird files on exoweb which end with .trans.bz!")

chunk_size =  300_000_000

if len(trans_filenames) > 0:
    for trans_filename in tqdm(trans_filenames):
        t_df[trans_filename]=pd.read_csv(trans_filename, compression='bz2',  
                                         sep='\s+',usecols=[0,1,2],header=None,  
                                         names=trans_col_name,chunksize= chunk_size,  
                                         iterator=True, low_memory=False)
else:
    raise IndexError("Cannot find .trans files on your device!")

print("Calculating ...")

data_new_fv_v1 = pd.DataFrame()
data_new_fv_v3 = pd.DataFrame()
#trans_Jmin = pd.DataFrame()
#trans_J_argmax = pd.DataFrame()

i_J_argmax = set(states_df.loc[states_df["J_i"]== J_argmax,'i'])
i_need = i_J_argmax #i_Jmin | i_J_argmax

for i in range(len(trans_filenames)):
    for chunk in t_df[trans_filenames[i]]:

        states_df_rename = states_df_complete.rename(columns={'i':'f'})
        states_df_rename = states_df_rename[['f']+labels]
        #trans_Jmin = trans_Jmin.append(chunk.loc[chunk["i"].isin(i_Jmin)])
        # chunk_Jmin = chunk.loc[chunk["i"].isin(i_Jmin)]
        # chunk_state_v1 =  pd.merge(chunk_Jmin, states_df_rename, on='f',how="left")
        # data_sum_fv_v1 = chunk_state_v1.groupby(['i']+labels).agg({'A_if':'sum'})
        # data_new_fv_v1 = data_new_fv_v1.append(data_sum_fv_v1)
        # data_new_fv_v1 = data_new_fv_v1.groupby(['i']+labels).agg({'A_if':'sum'})
        
        #trans_J_argmax  = trans_J_argmax.append(chunk.loc[chunk["i"].isin(i_J_argmax)])    
        chunk_J_argmax = chunk.loc[chunk["i"].isin(i_J_argmax)]
        chunk_state_v3 =  pd.merge(chunk_J_argmax, states_df_rename, on='f',how="left")
        data_sum_fv_v3 = chunk_state_v3.groupby(['i']+labels).agg({'A_if':'sum'})
        data_new_fv_v3 = data_new_fv_v3.append(data_sum_fv_v3)
        data_new_fv_v3 = data_new_fv_v3.groupby(['i']+labels).agg({'A_if':'sum'})



# data_new_fv_v1.rename(columns={'A_if':'sum_A_if'}, inplace = True)
# data_new_fv_v1 = data_new_fv_v1.reset_index()

data_new_fv_v3.rename(columns={'A_if':'sum_A_if'}, inplace = True)
data_new_fv_v3 = data_new_fv_v3.reset_index()

data_new = data_new_fv_v3.groupby(['i']).agg({'sum_A_if':'sum'}).reset_index()
data_new["lifetime_i"] = 1/data_new["sum_A_if"]

# Save intermediate data (optional)
data_new.to_csv('./original_result/'+molecule+"_origin_"+date+".csv")
#data_new_fv_v1.to_csv('./original_result/'+molecule+"_data_new_fv_v1_"+date+".csv",index = 0)
data_new_fv_v3.to_csv('./original_result/'+molecule+"_data_new_fv_v3_"+date+".csv",index = 0)


merged_result = pd.merge(states_df.loc[states_df["i"].isin(i_need)],data_new, on='i',how="left")
# manually set the lifetime of the ground state to be inf
merged_result.loc[merged_result.index ==0, 'lifetime_i'] = np.inf
merged_result = merged_result.dropna(subset = ['lifetime_i'])

## Final results: three methods (Method 2 will not be used)
# Method 1: only keep Jmin 
# Jmin = np.min(merged_result["J_i"])
# print("Result 1 using Jmin = ",Jmin)

# average on +/- and e/f
# merged_result_Jmin = merged_result[(merged_result["J_i"]==Jmin)]
# lifetime_avg_v1 =merged_result_Jmin.groupby(labels).agg({'lifetime_i':'mean'})
## generate csv files to store result version 1
# lifetime_avg_v1.to_csv('./v1_result/'+molecule+"_v1_"+date+".csv")

# Method 2: average on "labels" and +/- and e/f
# labels may be ("v") or ("v1","v2',"v3") or ("State","v","Lambda","Sigma","Omega"), etc.
#lifetime_avg_v2 =merged_result.groupby(labels).agg({'lifetime_i':'mean'})
#lifetime_avg_v2.to_csv('./v2_result/'+molecule+"_v2_"+date+".csv")

# Method 3: only keep J that maximizes P(J)
print("Result 3 using J that maximize P(J), which = ",J_argmax)
merged_result_J_argmax = merged_result[(merged_result["J_i"]==J_argmax)]
lifetime_avg_v3 = merged_result_J_argmax.groupby(labels).agg({'lifetime_i':'mean'})
lifetime_avg_v3 = lifetime_avg_v3.reset_index()
v_state = lifetime_avg_v3[labels]
v_state_vec = np.array(v_state)
v_state_vec = v_state_vec.tolist()
lifetime_avg_v3.insert(0,'v_state',v_state_vec)
lifetime_avg_v3.drop(labels, axis=1, inplace=True)
lifetime_avg_v3.rename(columns={'v_state':labels}, inplace = True)
lifetime_avg_v3.to_csv('./v3_result/'+molecule+"_v3_"+date+".csv",index = 0)


## Comparison among three methods
# merge1 = pd.merge(lifetime_avg_v1,lifetime_avg_v3,on=labels,how="outer")
#merge2 = pd.merge(merge1,lifetime_avg_v3,on=labels,how="outer")
# name1 = 'lifetime_v1,J='+ str(Jmin)
# name3 = 'lifetime_v3,J='+ str(J_argmax)
# merge1.rename(columns={'lifetime_i_x':name1,'lifetime_i_y':name3}, inplace = True)
# merge1.to_csv('./compare_result/'+molecule+"_lifetime_compare.csv")

## Decay lifetime 
initial_states = states_df[labels].drop_duplicates(subset = labels, keep='first')
#initial_states.drop(initial_states[initial_states.v ==0].index, inplace=True)


# # - method 1 (J = Jmin)
# states_df_Jmin = states_df.loc[states_df["J_i"]== Jmin]
# i_lists_Jmin = list(set(states_df_Jmin["i"]) & set(data_new_fv_v1["i"]))

def find_i(states_group):
    return list(states_group["i"])
    
# initial_ilists_v1 = states_df_Jmin[labels+['i']].groupby(labels).apply(find_i).reset_index()
# initial_ilists_v1.rename(columns={0:'i'}, inplace = True)

# decay_v1_df = pd.DataFrame()

# for i in range(len(initial_ilists_v1)):
#     initial_state = list(initial_ilists_v1.iloc[i][labels])
#     grouped_lifetime = data_new_fv_v1.loc[data_new_fv_v1["i"].isin(initial_ilists_v1.iloc[i]['i'])]
#     grouped_lifetime = grouped_lifetime.groupby(labels).agg({'sum_A_if':'sum'})
#     grouped_lifetime = grouped_lifetime.reset_index()
#     grouped_lifetime.insert(grouped_lifetime.shape[1], 'lifetime_i',1/grouped_lifetime["sum_A_if"])
#     # drop the records with v' = v"
#     drop_index = list(grouped_lifetime[labels][grouped_lifetime[labels] == initial_state].dropna(subset = labels).index)
#     grouped_lifetime = grouped_lifetime.drop(drop_index)
#     grouped_lifetime.insert(grouped_lifetime.shape[1], 'branching_ratio',grouped_lifetime['sum_A_if']/np.sum(grouped_lifetime['sum_A_if']))
#     grouped_lifetime = grouped_lifetime.sort_values(by= ['branching_ratio'],ascending = False)

#     # drop the record with branching_ratio < 0.0001
#     grouped_lifetime.drop(grouped_lifetime[grouped_lifetime.branching_ratio < 0.001].index, inplace=True)
#     # normalize the branching_ratio
#     grouped_lifetime['branching_ratio'] = grouped_lifetime['sum_A_if']/np.sum(grouped_lifetime['sum_A_if'])
#     grouped_lifetime.insert(0,'initial_state',str(initial_state))
#     decay_v1_df = decay_v1_df.append(grouped_lifetime)

#decay_v1_df.to_csv('./decay_result/'+molecule+'/v1'+'/'+molecule+".csv",index = 0)

# - method 3 (J = arg_Jmax)

states_df_J_argmax = states_df.loc[states_df["J_i"]== J_argmax]
i_lists_J_argmax = list(set(states_df_J_argmax["i"]) & set(data_new_fv_v3["i"]))

initial_states = states_df_J_argmax[labels].drop_duplicates(subset = labels, keep='first')
#need to improve: some might not exist in trans / no ground state

initial_ilists_v3 = states_df_J_argmax[labels+['i']].groupby(labels).apply(find_i).reset_index()
initial_ilists_v3.rename(columns={0:'i'}, inplace = True)

decay_v3_df = pd.DataFrame()
for i in range(len(initial_ilists_v3)):
    initial_state = list(initial_ilists_v3.iloc[i][labels])
    grouped_lifetime = data_new_fv_v3.loc[data_new_fv_v3["i"].isin(initial_ilists_v3.iloc[i]['i'])]
    grouped_lifetime = grouped_lifetime.groupby(labels).agg({'sum_A_if':'sum'})
    grouped_lifetime = grouped_lifetime.reset_index()
    grouped_lifetime.insert(grouped_lifetime.shape[1], 'lifetime_i',1/grouped_lifetime["sum_A_if"])
    # drop the records with v' = v"
    drop_index = list(grouped_lifetime[labels][grouped_lifetime[labels] == initial_state].dropna(subset = labels).index)
    grouped_lifetime = grouped_lifetime.drop(drop_index)
    grouped_lifetime.insert(grouped_lifetime.shape[1], 'branching_ratio',grouped_lifetime['sum_A_if']/np.sum(grouped_lifetime['sum_A_if']))
    grouped_lifetime = grouped_lifetime.sort_values(by= ['branching_ratio'],ascending = False)

    # drop the record with branching_ratio < 0.0001
    grouped_lifetime.drop(grouped_lifetime[grouped_lifetime.branching_ratio < 0.001].index, inplace=True)
    # normalize the branching_ratio
    grouped_lifetime['branching_ratio'] = grouped_lifetime['sum_A_if']/np.sum(grouped_lifetime['sum_A_if'])
    grouped_lifetime.insert(0,'initial_state',str(initial_state))
    decay_v3_df = decay_v3_df.append(grouped_lifetime)
    #grouped_lifetime.to_csv('./decay_result/'+molecule+'/v3'+'/'+molecule+str(initial_state)+".csv")

final_state = decay_v3_df[labels]
final_state_vec = np.array(final_state)
final_state_vec = final_state_vec.tolist()

decay_v3_df.insert(1,'final_state',final_state_vec)
decay_v3_df.drop(labels, axis=1, inplace=True)
decay_v3_df.rename(columns={'lifetime_i':'lifetime'}, inplace = True)
decay_v3_df.to_csv('./decay_result/'+molecule+'/v3'+'/'+molecule+'_'+date+".csv",index = 0)

end_time = time.time()
duration_ms = (end_time - start_time)/60
print('total time used is:',duration_ms,'min')

info_dic = {'molecule':[molecule],'labels':str(labels),'date':date,'chunk_size':chunk_size,'time (min)':duration_ms,'J':J_argmax,"n_trans":len(trans_filenames)}
info_df = pd.DataFrame(info_dic)
info_df.to_csv('./compute_info/'+molecule+".csv",index = 0,mode='a')
