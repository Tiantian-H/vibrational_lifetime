import glob
import pandas as pd
# no states file: CN,PS(POPS?),PO,SiH4,PH,trans-P2H2
## load infomation from def file 
iso_info = pd.read_pickle("molecule_first_iso_final.pickle")

## select which molecule to process
molecule_list = iso_info['molecule']
wrong_molecule = []
num = 0
num_from_states = []
num_from_def = []

for molecule in molecule_list:
    iso_slug = iso_info.loc[iso_info["molecule"]==molecule,"iso_slug"]
    linelist = iso_info.loc[iso_info["molecule"]==molecule,"linelist"]
    if molecule == "CN":
        linelist = 'Trihybrid'
    if molecule == 'trans-P2H2':
        continue
    path_mol_iso_list = list(molecule+'/'+iso_slug+'/'+linelist)
    if len(path_mol_iso_list) >0:
        path_mol_iso = path_mol_iso_list[0]
        read_path = '../../exomol/exomol3_data/'
        
        ## read states file
        # manually adjust headers due to incorrect def file
        # states_col_name = ['i','E_i','g_i','J_i','tau_i','Gamma','Ka','Kc','v1','v2','v3','e/c'] #H2S
        states_col_name = iso_info.loc[iso_info["molecule"]==molecule ,"headers"].values[0]    
        s_df = dict()
        states_df = pd.DataFrame()
        states_filenames = glob.glob(read_path + path_mol_iso + '/' + path_mol_iso.split('/')[1]+ '__' + path_mol_iso.split('/')[2] + '.states.bz2')
        try:
            states_filename = states_filenames[0]
            s_df[states_filename] =pd.read_csv(states_filename, compression='bz2', sep='\s+',
                                                    header=None, 
                                                    chunksize=100_000_000, iterator=True,
                                                    low_memory=False)
            
            col_num  = s_df[states_filename].get_chunk(1).shape[1]
    
                
            if col_num != len(states_col_name):
                wrong_molecule.append(molecule)
                num_from_def.append(len(states_col_name))
                num_from_states.append(col_num)
                
            num = num +1
            #print(num)
        except:
            print(read_path + path_mol_iso + '/' + path_mol_iso.split('/')[1]+ '__' + path_mol_iso.split('/')[2] + '.states.bz2')
    else:
        continue

diff_dict = {"molecule":wrong_molecule,"num_from_states":num_from_states,"num_from_def":num_from_def}
diff_molecule = pd.DataFrame(diff_dict)

diff_merged = pd.merge(diff_molecule,iso_info[["molecule","linelist","iso_slug","headers"]], on='molecule',how="left")
diff_merged.to_csv("diff_merged_.csv")

