#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import csv
import sqlite3
import time
import os
import math
import glob
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO

date = time.strftime("%d/%m/%Y")
date = date.replace('/','-')
start_time = time.time()

## load infomation from def file 
iso_info = pd.read_pickle("molecule_first_iso_final.pickle")

# ignore some molecules that cannot be processed at present
all_molecules = set(iso_info["molecule"])
diff_merged = pd.read_csv("diff_merged.csv")
wrong_molecules = set(diff_merged["molecule"])

correct_molecules = all_molecules - wrong_molecules # ignore the molecules with wrong headers
correct_molecules =  correct_molecules - {'trans-P2H2'} # data not available on the website
# ignore the molecules whose recommended linelist is MoLLIST since they are not complete which may lead to some very strange vibrational lifetimes.
MoLLIST = set(iso_info.loc[iso_info["linelist"]=='MoLLIST','molecule']) 
correct_molecules = correct_molecules - MoLLIST
correct_molecules = correct_molecules | {"CH","NH3","CO2","H3O_p","SiH","H3_p","SiO2","VO","SiH2","H2O","AlO","H2S"}

print("The following molecules can be processed:")
print(" ")
print(correct_molecules)
print(" ")

molecule = input("Please enter the name of the molecule:")
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

if os.path.exists('v1_result'):
    pass
else:
    os.makedirs('v1_result', exist_ok=True)

#if os.path.exists('v2_result'):
#    pass
#else:
#    os.makedirs('v2_result', exist_ok=True)

if os.path.exists('v3_result'):
    pass
else:
    os.makedirs('v3_result', exist_ok=True)

if os.path.exists('decay_result'):
    pass
else:
    os.makedirs('decay_result', exist_ok=True)
    
if os.path.exists('decay_result/'+ molecule+'/v1'):
    pass
else:
    os.makedirs('decay_result/'+ molecule+'/v1', exist_ok=True)
    
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
read_path = '../exomol/exomol3_data/'
#'../exomol/exomol3_data/'
# './www.exomol.com/db/'
#'../exomol/exomol3_data/'

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
    states_col_name = ['n','E_i','g_i','J_i','tau_i','g-factor','Parity','e/f','State','v','Lambda','Sigma','Omega']
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

#['i','E_i','g','J_i','tau_i','+/-','e/f','State','v','Lambda','Sigma','Omega']
#iso_info.loc[iso_info["molecule"]==molecule ,"headers"].values[0]
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
#if 'Lambda' in colnames:
#    labels = ["State",'v','Lambda','Sigma','Omega']
#    states_df.drop(states_df[states_df.v<0].index,inplace= True)

#elif 'v' in colnames:
#    labels = ['v']
#    states_df.drop(states_df[states_df.v<0].index,inplace= True)

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
            states_df[colnames[i]] = [np.float(x) for x in states_df[colnames[i]]]
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

connection = sqlite3.connect('tt.sqlite3')

cursor = connection.cursor()
cursor.execute('DROP TABLE IF EXISTS data;')
cursor.execute('CREATE TABLE data (i INTEGER, f INTEGER, A_if REAL);')

filenames = trans_filenames 
num_file = 0
for filename in filenames:
    with bz2.open(filename, 'rt') as bf:
        for line in bf:
            i, f, A_if = line.split()
            cursor.execute(f"INSERT INTO data (i, f, A_if) VALUES ({i}, {f}, {A_if});")
        connection.commit()
        num_file += 1
        #print(num_file)
connection.close()

data_new = pd.DataFrame()
connection = sqlite3.connect('tt.sqlite3')
#cursor = connection.cursor() 
data_new = pd.read_sql_query("select i, sum(A_if) from data group by i;",connection)
#for row in cursor.execute("select i, sum(A_if) from data group by i;"):
    #data_new.append(row) 
connection.close()
data_new.rename(columns={'sum(A_if)':'sum_A_if'}, inplace = True)
data_new["lifetime_i"] = 1/data_new["sum_A_if"]

print("Calculating ...")

#data_new_fv_v1 = pd.DataFrame()
i_J_argmax = set(states_df.loc[states_df["J_i"]== J_argmax,'i'])
data_new_fv_v3 = data_new.loc[data_new["i"].isin(i_J_argmax)]

i_need = i_J_argmax #i_Jmin | i_J_argmax
# Save intermediate data (optional)
#data_new.to_csv('./original_result/'+molecule+"_origin_"+date+".csv")
#data_new_fv_v1.to_csv('./original_result/'+molecule+"_data_new_fv_v1_"+date+".csv",index = 0)
#data_new_fv_v3.to_csv('./original_result/'+molecule+"_data_new_fv_v3_"+date+".csv",index = 0)


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
lifetime_avg_v3.to_csv('./v3_result/'+molecule+"_v3_sql_"+date+".csv",index = 0)

end_time = time.time()
duration_ms = (end_time - start_time)/60
print('total time used is:',duration_ms,'min')

info_dic = {'molecule':[molecule],'labels':str(labels),'date':date,'time (min)':duration_ms,'J':J_argmax,"n_trans":len(trans_filenames),"algorithm":"sql"}
info_df = pd.DataFrame(info_dic)
info_df.to_csv('./compute_info/'+molecule+"_algorithms.csv",index = 0,mode='a')