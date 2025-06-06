# ***** Utility functions to facilitate more robust modelling. *****
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
from pathlib import Path
import pandas as pd
from colored import fg, bg, attr
import tempfile
import shutil as sh
from filelock import FileLock as FL
import subprocess as sp
from datetime import datetime as DT
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.worksheet._read_only")

#-----------------------------------------------------------------------------------------------------------------------------------
Pre_Sign = f"{fg(52)}{'*'*80}{attr('reset')}\n\n"
Sign = f"{fg(52)}\nend_of_transmission\n{'*'*80}{attr('reset')}\n"
path_WS = 'C:/OD/WS_Mdl'
path_RunLog = PJ(path_WS, 'Mng/WS_RunLog.xlsx')
path_log = PJ(path_WS, 'Mng/log.csv')
## Can make get paths function that will provide the general directories, like path_WS, path_Mdl. Those can be derived from a folder structure.

# Get paths from MdlN --------------------------------------------------------------------------------------------------------------
def MdlN_Se_from_RunLog(MdlN): # Can be made faster. May need to make excel export the RunLog as a csv, so that I can use pd.read_csv instead of pd.read_excel. 
    """Returns RunLog line that corresponds to MdlN as a S."""
    DF = pd.read_excel(PJ(path_WS, 'Mng/WS_RunLog.xlsx'), sheet_name='RunLog')    
    Se_match = DF.loc[DF['MdlN'] == MdlN]
    if Se_match.empty:
        raise ValueError(f"MdlN {MdlN} not found in RunLog. {fg('indian_red_1c')}Check the spelling and try again.{attr('reset')}")
    S = Se_match.squeeze()
    return S

def paths_from_MdlN_Se(S, MdlN):
    """Takes in S, returns relevant paths."""
    Mdl, SimN_B = S[['model alias', 'B SimN']]
    MdlN_B = Mdl + str(SimN_B)
    
    d_path = {}
    d_path['Mdl']                   =   Mdl
    d_path['MdlN_B']                =   MdlN_B
    d_path['path_Mdl']              =   PJ(path_WS, f'models/{Mdl}')
    d_path['path_INI']              =   PJ(d_path['path_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.ini')
    d_path['path_BAT']              =   PJ(d_path['path_Mdl'], f'code/Mdl_Prep/Mdl_Prep_{MdlN}.bat')
    d_path['path_PRJ']              =   PJ(d_path['path_Mdl'], f'In/PRJ/{MdlN}.prj')
    d_path['path_Smk']              =   PJ(d_path['path_Mdl'], f'code/snakemake/{MdlN}.smk')
    d_path['path_Smk_temp']         =   PJ(d_path['path_Mdl'], f'code/snakemake/temp')
    d_path['path_MdlN']             =   PJ(d_path['path_Mdl'], f"Sim/{MdlN}")
    d_path['path_Out_HD']           =   PJ(d_path['path_MdlN'], f"GWF_1/MODELOUTPUT/HEAD/HEAD")
    d_path['path_PoP']              =   PJ(d_path['path_Mdl'], 'PoP')
    d_path['path_PoP_Out_MdlN']     =   PJ(d_path['path_PoP'], 'Out', MdlN)
    d_path['path_MM']               =   PJ(d_path['path_PoP_Out_MdlN'], f'MM-{MdlN}.qgz')
    d_path['path_INI_B']            =   d_path['path_INI'].replace(MdlN, MdlN_B)
    d_path['path_BAT_B']            =   d_path['path_BAT'].replace(MdlN, MdlN_B)
    d_path['path_PRJ_B']            =   d_path['path_PRJ'].replace(MdlN, MdlN_B)
    d_path['path_Smk_B']            =   d_path['path_Smk'].replace(MdlN, MdlN_B)
    d_path['path_MdlN_B']           =   d_path['path_MdlN'].replace(MdlN, MdlN_B)
    d_path['path_Out_HD_B']         =   d_path['path_Out_HD'].replace(MdlN, MdlN_B)
    d_path['path_PoP_Out_MdlN_B']   =   d_path['path_PoP_Out_MdlN'].replace(MdlN, MdlN_B)
    d_path['path_MM_B']             =   d_path['path_MM'].replace(MdlN, MdlN_B)

    return  d_path

def get_MdlN_paths(MdlN: str, verbose=False): #666 Can be split into two as both S and B aren't allways needed. Or better, I can make a new function that does that for just 1 run.
    """ Returns a dictionary of useful object (MdlN_B, directories etc.) for a given model. Those need to then be passed to arguments, e.g. path_INI_B = Dft_paths['path_INI_N']."""
    d_paths = paths_from_MdlN_Se( MdlN_Se_from_RunLog((MdlN)), MdlN )
    if verbose:
        print(f"🟢 - {MdlN} paths extracted from RunLog and returned as dictionary with keys:\n{', '.join(d_paths.keys())}")
    return d_paths
# ----------------------------------------------------------------------------------------------------------------------------------

# READ FILES -----------------------------------------------------------------------------------------------------------------------
#666 to be iproved later by replacing paths with MdlN. I'll have to make get_MdlN_paths_noB, where RunLog won't be read. Path of one MdlN will be calculated off of standard folder structure.
def read_IPF_Spa(path_IPF):
    """Reads IPF file without temporal component - i.e. no linked TS text files. Returns a DF created from just the IPF file.""" 
    with open(path_IPF, "r") as f:
        l_Ln = f.readlines()

    N_C = int(l_Ln[1].strip())  # Number of columns
    l_C_Nm = [l_Ln[I + 2].split("\n")[0] for I in range(N_C)] # Extract column names
    DF_IPF = pd.read_csv(path_IPF, skiprows=2+N_C+1, names=l_C_Nm)

    print(f"🟢 - IPF file {path_IPF} read successfully. DataFrame created with {len(DF_IPF)} rows and {len(DF_IPF.columns)} columns.")
    return DF_IPF

def INI_to_d(path_INI:str) -> dict:
    """Reads INI file (used for preparing the model) and converts it to a dictionary. Keys are converted to upper-case.
    Common use:
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE']
    N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ), 
    print(f'The model area has {N_R} rows and {N_C} columns.'))
    """
    d_INI = {}
    with open(path_INI, 'r', encoding="utf-8") as file:
        for l in file:
            l = l.strip()
            if l and not l.startswith("#"):  # Ignore empty lines and comments
                k, v = l.split("=", 1)  # Split at the first '='
                d_INI[k.strip().upper()] = v.strip()  # Remove extra spaces
    
    print(f"🟢 - INI file {path_INI} read successfully. Dictionary created with {len(d_INI)} keys.")
    return d_INI

def Mdl_Dmns_from_INI(path_INI): # 666 Can be improved. It should take a MdlN instead of a path. Makes things easier.
    """Returns model dimension parameters. Common use:
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = WS.Mdl_Dmns_from_INI(path)"""
    d_INI = INI_to_d(path_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize )

    print(f"🟢 - model dimensions extracted from {path_INI}.")
    return Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C

def Sim_Cfg(*l_MdlN, path_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    print(f"\n{'-'*100}\nOpening all configuration files for specified runs with the default program.\nIt's assumed that Notepad++ is installed in: {path_NP}.\nIf false, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n")
    
    l_keys = ['path_Smk', 'path_BAT', 'path_INI', 'path_PRJ']
    l_paths = [get_MdlN_paths(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]
    sp.Popen([path_NP] + l_files)
    for f in l_files:
        print(f'🟢 - {f}')

def HD_Out_IDF_to_DF(path): #666 can make it save DF if a 2nd path is provided. Unecessary for now.
    """Reads IDF files from the given path and returns a DataFrame with the file names and their corresponding parameters. Parameters are extracted from filnames, based on a standard format. Hence, don't use this for other groups of IDF files, unless you're sure they follow the same format.""" #666 can be generalized later, to work on all sorts of IDF files.
    Se_Fi_path = pd.Series([PJ(path, i) for i in LD(path) if i.lower().endswith('.idf')])
    DF = pd.DataFrame({'path':Se_Fi_path, 'file': Se_Fi_path.apply(lambda x: PBN(x))})
    DF[['type', 'year', 'month', 'day', 'L']] = DF['file'].str.extract(
        r'^(?P<type>[A-Z]+)_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\d{6}_L(?P<L>\d+)\.IDF$'
        ).astype({'year': int, 'month': int, 'day': int, 'L': int})
    # DF.to_csv(PJ(path, 'contents.csv'), index=False)
    return DF
# ----------------------------------------------------------------------------------------------------------------------------------

# Sim Prep + Run -------------------------------------------------------------------------------------------------------------------
def S_from_B(MdlN:str):
    """Copies files that contain Sim options from the B Sim, renames them for the S Sim, and opens them in the default file editor. Assumes default WS_Mdl folder structure (as described in READ_ME.MD)."""    
    print(Pre_Sign)
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    MdlN_B, path_INI_B, path_INI, path_BAT_B, path_BAT, path_Smk, path_Smk_B, path_PRJ_B, path_PRJ = (d_paths[k] for k in ['MdlN_B', "path_INI_B", "path_INI", "path_BAT_B", "path_BAT", "path_Smk", "path_Smk_B", "path_PRJ_B", "path_PRJ"]) # and pass them to objects that will be used in the function

    # Copy .INI, .bat, .prj and make default (those apply to every Sim) modifications
    for path_B, path_S in zip([path_Smk_B, path_BAT_B, path_INI_B], [path_Smk, path_BAT, path_INI]):
        try:
            if not os.path.exists(path_S): # Replace the MdlN of with the new one, so that we don't have to do it manually.
                sh.copy2(path_B, path_S)
                with open(path_S, 'r') as f1:
                    contents = f1.read()
                with open(path_S, 'w') as f2:
                    f2.write(contents.replace(MdlN_B, MdlN))
                if ".bat" not in path_B.lower():
                    os.startfile(path_S) # Then we'll open it to make any other changes we want to make. Except if it's the BAT file
                print(f"🟢 - {path_S.split('/')[-1]} created successfully! (from {path_B})")
            else:
                print(f"\u274C - {path_S.split('/')[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.")
        except Exception as e:
            print(f"\u274C - Error copying {path_B} to {path_S}: {e}")

    try:
        if not os.path.exists(path_PRJ): # For the PRJ file, there is no default text replacement to be performed, so we'll just copy.
            sh.copy2(path_PRJ_B, path_PRJ)
            os.startfile(path_PRJ) # Then we'll open it to make any other changes we want to make.
            print(f"🟢 - {path_PRJ.split('/')[-1]} created successfully! (from {path_PRJ_B})")
        else:
            print(f"\u274C - {path_PRJ.split('/')[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.")
    except Exception as e:
        print(f"\u274C - Error copying {path_PRJ_B} to {path_PRJ}: {e}")
    print(Sign)

def S_from_B_undo(MdlN:str):
    """Will undo S_from_B by deletting S files"""
    print(Pre_Sign)
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    MdlN_B, path_INI_B, path_INI, path_BAT_B, path_BAT, path_Smk, path_Smk_B, path_PRJ_B, path_PRJ = (d_paths[k] for k in ['MdlN_B', "path_INI_B", "path_INI", "path_BAT_B", "path_BAT", "path_Smk", "path_Smk_B", "path_PRJ_B", "path_PRJ"]) # and pass them to objects that will be used in the function

    confirm = input(f"Are you sure you want to delete the Cfg files (.smk, .ini, .bat, .prj) for {MdlN}? (y/n): ").strip().lower()
    if confirm == 'y':
        for path_S in [path_Smk, path_BAT, path_INI, path_PRJ]:
            os.remove(path_S) # Delete the S files
            print(f'🟢 - {path_S.split("/")[-1]} deleted successfully!')
    print(Sign)

def Up_log(MdlN: str, d_Up: dict, path_log=PJ(path_WS, 'Mng/log.csv')):
    """Update log.csv based on MdlN and key of `updates`."""
    path_lock = path_log + '.lock'  # Create a lock file to prevent concurrent access
    lock = FL(path_lock)

    with lock:  # Acquire the lock to prevent concurrent access
        DF = pd.read_csv(path_log, index_col=0)  # Assumes log.csv exists.

        for key, value in d_Up.items():  # Update the relevant cells
            DF.at[MdlN, key] = value

        while True: # Wait for file to be closed if it's open
            try:
                DF.to_csv(path_log, date_format='%Y-%m-%d %H:%M')  # Save back to CSV
                break  # Break if successful
            except PermissionError:
                input("log.csv is open. Press Enter after closing the file...")  # Wait for user input

def RunMng(cores=None, DAG:bool=True):
    """Read the RunLog, and for each queued model, run the corresponding Snakemake file."""
    if cores is None:
        cores = max(cpu_count() - 2, 1) # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    print(f"{Pre_Sign}RunMng will run all Sims that are queued in the RunLog.\n")

    print(f"--- Reading RunLog ...", end='')
    DF = pd.read_excel(path_RunLog, sheet_name='RunLog').dropna(subset='runN') # Read RunLog
    DF_q = DF.loc[ ((DF['Start Status'] == 'Queued') & ((DF['End Status'].isna()) | (DF['End Status']=='Failed')))  |
                   ((DF['Start Status'] == 'Re-queued'))] # _q for queued. Only Run Queued runs that aren't running or have finished.
    print(' completed!\n')

    print('--- Running snakemake files:')
    if DF_q.empty:
        print("\n🔴 - No queued runs found in the RunLog.")
    else:
        for i, Se_Ln in DF_q.iterrows():
            path_Smk = PJ(path_WS, f"models/{Se_Ln['model alias']}/code/snakemake/{Se_Ln['MdlN']}.smk")
            path_log = PJ(path_WS, f"models/{Se_Ln['model alias']}/code/snakemake/log/{Se_Ln['MdlN']}_{DT.now().strftime('%Y%m%d_%H%M%S')}.log")
            path_DAG = PJ(path_WS, f"models/{Se_Ln['model alias']}/code/snakemake/DAG/DAG_{Se_Ln['MdlN']}.png")
            print(f" -- {fg('green')}{PBN(path_Smk)}{attr('reset')}\n")

            try:
                if DAG:
                    sp.run(["snakemake", "--dag", "-s", path_Smk, "--cores", str(cores), '|', 'dot', '-Tpng', '-o', f'{path_DAG}'], shell=True, check=True)
                with open(path_log, 'w') as f:
                    sp.run(["snakemake", "-p", "-s", path_Smk, "--cores", str(cores)], check=True, stdout=f, stderr=f) # Run snakemake and write output to log file
                print(f"🟢")
            except sp.CalledProcessError as e:
                print(f"🔴: {e}")
    print(Sign)

def reset_Sim(MdlN: str):
    """
    Resets the simulation by:
        1. Deleting all files in the MldN folder in the Sim folder.
        2. Clearing log.csv.
        3. Deletes Smk log files for MdlN.
        4. Deletes PoP folder for MdlN.
    """
    
    permission = input(f"This will delete the Sim/{MdlN} folder and clear the corresponding line of the log.csv. Are you sure you want to proceed? (y/n): ").strip().lower()
    print(f"{Pre_Sign}Resetting the simulation for {MdlN}.\n")
    
    if permission == 'y':
        d_paths = get_MdlN_paths(MdlN) # Get default directories
        path_MdlN = d_paths['path_MdlN']
        DF = pd.read_csv(path_log) # Read the log file
        path_Smk_temp = d_paths['path_Smk_temp']
        l_temp = [i for i in LD(path_Smk_temp) if MdlN in i]

        if os.path.exists(path_MdlN) or (MdlN in DF['MdlN'].values) or l_temp or os.path.exists(d_paths['path_PoP_Out_MdlN']): # Check if the Sim folder exists or if the MdlN is in the log file
            i = 0

            try:
                if not os.path.exists(path_MdlN):
                    raise FileNotFoundError(f"{path_MdlN} does not exist.")
                sp.run(f'rmdir /S /Q "{path_MdlN}"', shell=True) # Delete the entire Sim folder
                print(f"🟢 - Sim folder removed successfully.")
                i += 1
            except:
                print(f"🔴 - failed to delete Sim folder.")

            try:
                DF[ DF['MdlN']!=MdlN ].to_csv(path_log, index=False) # Remove the log entry for this model
                print("🟢 - Log file updated successfully.")
                i += 1
            except:
                print(f"🔴 - failed to update log file.")
            
            try:
                print(path_Smk_temp)
                for l in l_temp:
                    os.remove(PJ(path_Smk_temp, l))
                print("🟢 - Smk log files deleted successfully.")
                i += 1
            except:
                print(f"🔴 - failed to remove Smk log files.")

            try:
                if not os.path.exists(d_paths['path_PoP_Out_MdlN']):
                    raise FileNotFoundError(f"{d_paths['path_PoP_Out_MdlN']} does not exist.")
                sp.run(f'rmdir /S /Q "{d_paths['path_PoP_Out_MdlN']}"', shell=True) # Delete the entire Sim folder
                print(f"🟢 - PoP Out folder removed successfully.")
                i += 1
            except:
                print(f"🔴 - failed to delete PoP Out folder.")

            if i==4:
                print("\n🟢 - ALL files were successfully removed.")
            else:
                print(f"🟡 - {i}/4 sub-processes finished successfully.")
        else:
            print(f"🔴 - Items do not exist (Sim folder, log entry, Smk log files, PoP Out folder). No need to reset.")
    else:
        print(f"🔴 - Reset cancelled by user (you).")
    print(Sign)

def get_elapsed_time_str(start_time: float) -> str:
    s = int((DT.now() - start_time).total_seconds())
    d, h, m, s = s // 86400, (s // 3600) % 24, (s // 60) % 60, s % 60

    if d:   return f"{d}-{h:02}:{m:02}:{s:02}"
    return f"{h:02}:{m:02}:{s:02}"
#-----------------------------------------------------------------------------------------------------------------------------------

def get_last_MdlN():
    path_log = PJ(path_WS, 'Mng/log.csv')
    DF = pd.read_csv(path_log)
    DF.loc[:-2, 'Sim end DT'] = DF.loc[:-2, 'Sim end DT'].apply(pd.to_datetime, dayfirst=True)
    DF['Sim end DT'] = pd.to_datetime(DF['Sim end DT'], dayfirst=True)
    return  DF.sort_values('Sim end DT', ascending=False).iloc[0]['MdlN']
