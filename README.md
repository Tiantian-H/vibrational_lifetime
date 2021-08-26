# A database of vibrational state lifetime

UCL MSc Scientific and Data Intensive Computing Research Project 

By Tiantian He


## Purpose

The project has constructed a new database that contains the vibrational state lifetimes as well as the partial lifetimes decaying from each initial state to their final states. This has been done using the molecule line lists extracted from the ExoMol database.

## Important Notes

To guarantee that this code can be implemented successfully, you should guarantee that:

1.You should use the exoweb, becuase the programming is based on the assumption that all the data from the ExoMol database have already existed. (But it should be fine if you download all the data from the ExoMol database locally on your own device and put them in the exactly same structure as that on the exoweb).

2.No matter you clone the repository from the GitHub by yourself, or use the repository that already exists in the file called lifetime_example_file on the exoweb, you should guarantee that you are in the correct path:
```bash
/mnt/data/<working_file>/vibrational_lifetime
```
Where "working_file" should be your own file or the file called lifetime_example_file that already exists. "vibrational_lifetime" is the name of this GitHub repository.

This is to guarantee that the relative path in the code is correct. The detailed steps of doing this can be found later in this documentation.

3.If the file/data name or struture on the ExoMol database or the exoweb changes in the future, the implementation of the code may be affected. In this case, please contact the autor of this code to do some modification accordingly.

4.Some molecules whose headers of the states files extracted from the def file are not consistent with the actual states file, so they are temporarily ignored. But it's possible to manually correct their headers within two simple steps. This will be described in the last part of this documentation.

5.For molecules whose energy at the ground state is not 0, you will not be able to obtain the result. Instead, traceback will be returned pointing out that the ground state energy is not 0.

## Environment

```bash
The exoweb websever
Python 3
```
## Connection to the exoweb
In the terminal, use the following ssh command to connect to the exoweb. Access is from within the UCL network only, so if you're outsibe the UCL compus please connect to the UCL VPN first.
```bash
ssh <username>@exoweb.projects.phys.ucl.ac.uk
```
Enter the password for authorization.

## Create your working directory and clone the repository from GitHub 
(Usage for the first time only)
```bash
cd /mnt/data
mkdir <working_file>
cd <working_file>
git clone https://github.com/Tiantian-H/vibrational_lifetime.git
```
## Alternatively : directly use the file called lifetime_example_file on the exoweb
There's already a copy of the above-mentioned repository on the exoweb. You can use it instead of making a new clone. In this case, <working_file> equals to lifetime_example_file.

## Install Dependancies

(Usage for the first time only)

Please use the following Linux commands to install the packages needed for this project if they haven't been installed before.

Take the library called glob as an example.

```bash
[<username>@exoweb vibrational_lifetime]$ python3 -m pip install --user glob
```

## Usage Instructions

Suppose your current working directory on the exoweb is:

/mnt/data/<working_file>/vibrational_lifetime

Here vibrational_lifetime is the name of the GitHub repository you have just created.

The original file structure without any outputs is:
```
.
├── README.md
├── linelist.csv
├── def_processing.py
├── diff_merged_utf8.csv
├── calculations.py
├── nohup_calculations.py
├── find_diff.py
├── total_life_three_versions.py
├── alternative_byline.py
├── alternative_sql.py
```
The explanations of these files are:

1.linelist.csv

A table contains which iso_slug and linelist to use for each molecule. (For some molecules, it's hard to automatically distiguish the information of the recommended linelist from the master file since it doesn't appear firstly in the master file. That's why a manually adjusted version is provided here.)

2.def_processing.py

This piece of code reads the information of molecules from linelist.csv to obtain the corresponding recommended line list for each molecule. Then it processes the def file for each molecule to extract useful information like the headers of the states file, etc. The output file is molecule_first_iso_final.csv and molecule_first_iso_final.pickle.

3.find_diff.py (Optional)

This piece of code finds out the molecules whose headers in the actual states file are not consistent with those extracted from the def file. The output file is diff_merged_utf8.csv. However, since diff_merged_utf8.csv has already been genrated in advance and exists in the repository, this code can be skipped.

4.calculations.py/ nohup_calculations.py

This is the main code for calculating the total lifetime and partial lifetime. There are two versions. You can use either of them.

Interactive version: calculations.py
You need to enter the name of the molecule that you want to calculate when you run the code each time. You can get some feedback in real time from your terminal. 

Non-hang-up version: nohup_calculations.py

The output files of calculations.py or nohup_calculations.py are:

a.compute_info/molecule: 
This file contains some useful information for each calculation, like the date, the computing time, the vibrational state, the J value that maximizes the Boltzmann function P(J).

b.decay_result/molecule/v3/molecule_date.csv:
The decay lifetime and braching ratio.

c.v3_result/molecule_v3_date:
The total lifetime.

Please use the following instructions to obtain the result.

These are the main files needed for calculations. Apart from these, their are also some alternative files which calculate other version of the total lifetime or process the trans files using other methods. These codes are used in the discussion part of the report. They are not necessarily to be run.

###  Step 1 Process the def files

You can skip this step, since the output file of this step (molecule_first_iso_final.pickle) has already been generated for you. 

```bash
[<username>@exoweb vibrational_lifetime]$
cd /mnt/data/<working_file>/vibrational_lifetime
[<username>@exoweb vibrational_lifetime]$ chmod u+x def_processing.py
[<username>@exoweb vibrational_lifetime]$ python3 def_processing.py
```
###  Step 2 Calculate the lifetimes
You will be able to see the list of all the molecules that you can calculate. You will be asked to enter one molecule that you want to calculate each time.

```bash
[<username>@exoweb vibrational_lifetime]$ chmod u+x calculations.py
[<username>@exoweb vibrational_lifetime]$ python3 calculations.py
```
For some larger molecules, you may want to keep the code running when you log out from the exoweb. However, you need to manually enter the molecule that you want to calculate by editing the code of nohup_calculations.py using:

```bash
[<username>@exoweb vibrational_lifetime]$ nano nohup_calculations.py 
```
When you open the code file using nano command, you can file the following position to enter the molecule name at the very beginning:

```bash
#############  choose which molecule to be processed #############
molecule = "SO3"
##################################################################
```
The default molecule here is SO3.
You need to save your modification by pressing control+O and then exit by pressing control+X.

Then, you can run nohup_calculations.py using

```bash
[<username>@exoweb vibrational_lifetime]$ mkdir log
[<username>@exoweb vibrational_lifetime]$ chmod u+x nohup_calculations.py
[<username>@exoweb vibrational_lifetime]$ nohup python3 nohup_calculations.py > ./log/nohup_calculations.log 2>&1 &
```

After the above-mentioned two steps, the file structure of this repository is (including the example outputs of some molecules):

```
.
├── README.md
├── linelist.csv
├── def_processing.py
├── molecule_first_iso_final.csv
├── molecule_first_iso_final.pickle
├── diff_merged_utf8.csv
├── calculations.py
├── nohup_calculations.py
├── find_diff.py
├── total_life_three_versions.py
├── alternative_byline.py
├── alternative_sql.py
├── compute_info
│   ├── AlO.csv
│   ├── CH4.csv
│   ├── CS.csv
│   ├── H2CO.csv
│   ├── H2O.csv
│   ├── H2O2.csv
│   ├── H2_p.csv
│   ├── H3_p.csv
│   ├── HF.csv
│   ├── HNO3.csv
│   ├── KCl.csv
│   ├── NH3.csv
│   ├── NaCl.csv
│   ├── NaH.csv
│   ├── PN.csv
│   ├── SO3.csv
│   ├── ScH.csv
│   ├── SiH.csv
│   ├── SiH2.csv
│   ├── SiH4.csv
│   ├── SiO.csv
│   ├── SiO2.csv
│   ├── SiO_algorithms.csv
│   ├── SiS.csv
│   └── VO.csv
├── decay_result
│   ├── AlO
│   │   └── v3
│   │       └── AlO_16-08-2021.csv
│   ├── CH4
│   │   └── v3
│   │       └── CH4_17-08-2021.csv
│   ├── CS
│   │   └── v3
│   │       ├── CS_23-08-2021.csv
│   │       └── CS_24-08-2021.csv
│   ├── H2CO
│   │   └── v3
│   │       └── H2CO_24-08-2021.csv
│   ├── H2O
│   │   └── v3
│   │       └── H2O_16-08-2021.csv
│   ├── H2O2
│   │   └── v3
│   │       └── H2O2_16-08-2021.csv
│   ├── H2_p
│   │   └── v3
│   │       └── H2_p_24-08-2021.csv
│   ├── H3_p
│   │   └── v3
│   │       └── H3_p_24-08-2021.csv
│   ├── HF
│   │   └── v3
│   │       └── HF_24-08-2021.csv
│   ├── HNO3
│   │   └── v3
│   │       └── HNO3_16-08-2021.csv
│   ├── KCl
│   │   └── v3
│   │       └── KCl_23-08-2021.csv
│   ├── NH3
│   │   └── v3
│   │       └── NH3_20-08-2021.csv
│   ├── NaCl
│   │   └── v3
│   │       └── NaCl_23-08-2021.csv
│   ├── NaH
│   │   └── v3
│   │       └── NaH_23-08-2021.csv
│   ├── PN
│   │   └── v3
│   │       ├── PN_23-08-2021.csv
│   │       └── PN_24-08-2021.csv
│   ├── SO3
│   │   └── v3
│   │       └── SO3_24-08-2021.csv
│   ├── ScH
│   │   └── v3
│   │       └── ScH_23-08-2021.csv
│   ├── SiH
│   │   └── v3
│   │       └── SiH_24-08-2021.csv
│   ├── SiH2
│   │   └── v3
│   │       ├── SiH2_11-08-2021.csv
│   │       └── SiH2_23-08-2021.csv
│   ├── SiH4
│   │   └── v3
│   │       └── SiH4_21-08-2021.csv
│   ├── SiO
│   │   └── v3
│   │       ├── SiO_16-08-2021.csv
│   │       └── SiO_23-08-2021.csv
│   ├── SiO2
│   │   └── v3
│   │       └── SiO2_10-08-2021.csv
│   ├── SiS
│   │   └── v3
│   │       └── SiS_24-08-2021.csv
│   └── VO
│       └── v3
│           └── VO_16-08-2021.csv
└── v3_result
    ├── AlO_v3_16-08-2021.csv
    ├── CH4_v3_17-08-2021.csv
    ├── CS_v3_23-08-2021.csv
    ├── CS_v3_24-08-2021.csv
    ├── H2CO_v3_24-08-2021.csv
    ├── H2O2_v3_16-08-2021.csv
    ├── H2O_v3_16-08-2021.csv
    ├── H2_p_v3_24-08-2021.csv
    ├── H3_p_v3_24-08-2021.csv
    ├── HF_v3_24-08-2021.csv
    ├── HNO3_v3_16-08-2021.csv
    ├── KCl_v3_23-08-2021.csv
    ├── NH3_v3_20-08-2021.csv
    ├── NaCl_v3_23-08-2021.csv
    ├── NaH_v3_23-08-2021.csv
    ├── PN_v3_23-08-2021.csv
    ├── PN_v3_24-08-2021.csv
    ├── SO3_v3_24-08-2021.csv
    ├── ScH_v3_23-08-2021.csv
    ├── SiH2_v3_11-08-2021.csv
    ├── SiH2_v3_23-08-2021.csv
    ├── SiH4_v3_21-08-2021.csv
    ├── SiH_v3_24-08-2021.csv
    ├── SiO2_v3_10-08-2021.csv
    ├── SiO_v3_16-08-2021.csv
    ├── SiO_v3_23-08-2021.csv
    ├── SiO_v3_byline_23-08-2021.csv
    ├── SiO_v3_sql_24-08-2021.csv
    ├── SiS_v3_24-08-2021.csv
    └── VO_v3_16-08-2021.csv

51 directories, 95 files

```
Finally, there are some alternative codes used in the discussion part of the report. It's not uncessary to run these codes to get the intended results.

1. total_life_three_versions.py:
This gives three versions of the total lifetimes.

2. alternative_byline.py
This process the trans files line by line to obtain the total lifetimes.

3. alternative_sql.py
This uses sqlit3 to obtain  the total lifetimes.

## For those molecules with inaccurate headers:
You can manually modify the code calculations.py and nohup_calculations.py following:

1.Change the code:
```bash
correct_molecules = correct_molecules | {"CH","NH3","CO2","H3O_p","SiH","H3_p","SiO2","VO","SiH2","H2O","AlO","H2S"}
```
into 
```bash
correct_molecules = correct_molecules | {"your molecule","CH","NH3","CO2","H3O_p","SiH","H3_p","SiO2","VO","SiH2","H2O","AlO","H2S"}
```
Where "your molecule" is the molecule you want to add. 
In fact , molecules like "CH","NH3","CO2"... here are also molecules with inaccurate headers that I manually added before.

Manually add the headers of this molecule at the place after this comment:
``` bash
# manually correct headers for some molecules whose headers from the def files are not right
```
For instance, for NaH, you should add:
```bash
if molecule == "NaH":
    states_col_name = ['i', 'E_i', 'g_i', 'J_i', 'tau_i','State','v']
```
Note that the first four elements must be 'i', 'E_i', 'g_i', 'J_i'.