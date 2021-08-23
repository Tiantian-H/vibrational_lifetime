#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# given that the def files for each molecule already exist on exoweb
import pandas as pd
import numpy as np
import glob

molecule_df_first_iso = pd.read_csv("linelist.csv")
# Read Def File of the first iso for each molecule (78 in total)
def_url = []#pd.DataFrame()
def_num = len(molecule_df_first_iso) 
molecule_list = molecule_df_first_iso["molecule"].values
iso_slug_list = molecule_df_first_iso["iso_slug"].values
isotopologue_list = molecule_df_first_iso["linelist"].values


# Extract the def files 
#path = "./data/def"
def_col_name = ['c0', '#', 'c1', 'c2', 'c3', 'c4', 'c5']
truncate_E = []
# headers = basic_headers (+ tau_i, if lifetime exists) + rest_headers
basic_headers = ['i','E_i','g_i','J_i']
rest_headers = []
#exist_life_pos = np.zeros(len(files))#np.zeros(len(molecule_df_first_iso))
filename_def = []
trans_lines_num_total = []
trans_files_num = []
states_lines_num = []

life_def_filename = []
no_life_def_filename = []

tot = 0
count = 0
def_molecule = []
exist_life_pos = []

for molecule in molecule_list:

    iso_slug = molecule_df_first_iso.loc[molecule_df_first_iso["molecule"]==molecule,"iso_slug"]
    linelist = molecule_df_first_iso.loc[molecule_df_first_iso["molecule"]==molecule,"linelist"]
    path_mol_iso_list = list(molecule+'/'+iso_slug+'/'+linelist)
    path_mol_iso = path_mol_iso_list[0]
    read_path = '../../exomol/exomol3_data/'
    def_filename = glob.glob(read_path + path_mol_iso+"/"+"*.def")
    if len(def_filename) > 1:
        print ("Warning: There are ",len(def_filename),"def files for",molecule,"!")
    
    try:
        def_df = pd.read_csv(def_filename[0],sep='\s+', names=def_col_name, header=None)
        def_molecule.append(molecule)
        tot += 1
        filename_def.append(def_filename)
        c1 = def_df['c1']
        if def_df[c1.isin(['Lifetime'])]['c0'].values == '1':
            life_def_filename.append(def_filename)
            count += 1 
            #exist_life_pos[tot-1] = 1
            exist_life_pos.append(1)
        else:
            exist_life_pos.append(1)
            no_life_def_filename.append(def_filename)
        E = list(def_df.loc[def_df['c2']=="energy",'c0'])
        if E == []:
            E = np.nan
        else:
            E = np.float(E[0])
        truncate_E.append(E)
            
        trans_line = def_df.loc[def_df['c4']=="transitions",'c0']
        if len(trans_line)==0:
            trans_line = np.nan
        else:
            trans_line = int(trans_line)
        trans_lines_num_total.append(trans_line)
            
        trans_num = def_df.loc[def_df['c3']=="transition",'c0']
        if len(trans_num) == 0:
            trans_num = np.nan
        else:
            trans_num = int(trans_num)
        trans_files_num.append(trans_num)
            
        states_line = def_df.loc[def_df['c3']=="states",'c0']
        if len(states_line) == 0:
            states_line = np.nan
        else:
            states_line = int(np.float(states_line.values[0]))
        states_lines_num.append(states_line)
            
        rest_headers.append(list(def_df.loc[(def_df['c1']=="Quantum") & (def_df['c2']=="label"),'c0']))
    except:  
        print ("No def file founded for",molecule,"!")

print('There are altogether ', tot, ' def files.\n')
print('There are', count, 'def files with lifetime availability = 1')


# def_molecule = []
# for name in filename_def:
#     if name.split("_")[1] == "p":
#         def_molecule.append(name.split("_")[0]+"_"+name.split("_")[1])
#     else:
#         def_molecule.append(name.split("_")[0])
    
for i in range(len(rest_headers)):
    if exist_life_pos[i] ==1:
        rest_headers[i].insert(0,"tau_i")
        rest_headers[i] = basic_headers + rest_headers[i]
    else:
        rest_headers[i] = basic_headers + rest_headers[i]


def_info = {'molecule':def_molecule,'truncate_E':truncate_E,'life':exist_life_pos,'headers':rest_headers,'trans_lines_num_total':trans_lines_num_total,'trans_files_num':trans_files_num,'states_lines_num':states_lines_num}
def_info = pd.DataFrame(def_info)
molecule_first_iso_final = pd.merge(molecule_df_first_iso, def_info, on='molecule',how='inner')

molecule_first_iso_final.to_csv('molecule_first_iso_final.csv')
molecule_first_iso_final.to_pickle('molecule_first_iso_final.pickle')
molecule_first_iso_final.to_csv('molecule_final.csv')


