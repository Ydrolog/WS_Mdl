# ***** Similar to utils.py, but those utilize imod, which takes a long time to load. *****
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import tempfile
import imod
from .utils import Sign, Pre_Sign, read_IPF_Spa, INI_to_d, get_MdlN_paths, Pa_WS, vprint
from . import utils as U
import numpy as np
import subprocess as sp
from multiprocessing import Process, cpu_count
import re
import pandas as pd
from tqdm import tqdm # Track progress of the loop
from filelock import FileLock as FL

# PRJ related ----------------------------------------------------------------------------------------------------------------------
def read_PRJ_with_OBS(Pa_PRJ, verbose:bool=True):
    """imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines."""
    with open(Pa_PRJ, "r") as f:
        lines = f.readlines()

    l_filtered_Lns, l_OBS_Lns = [], []
    skip_block = False

    for line in lines:
        if "(obs)" in line.lower():  # Start of OBS block
            skip_block = True
            l_OBS_Lns.append(line)  # Keep the header
        elif skip_block and line.strip() == "":  # End of OBS block
            skip_block = False
        elif skip_block:
            l_OBS_Lns.append(line)  # Store OBS content
        else:
            l_filtered_Lns.append(line)  # Keep everything else

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".prj") as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    PRJ = imod.formats.prj.read_projectfile(Pa_PRJ_temp) # Load the PRJ file without OBS
    os.remove(Pa_PRJ_temp) # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns

def PRJ_to_DF(MdlN):#, verbose:bool=True): #666 adding verbose behaviour enables print surpression when verbose=False
    """Leverages read_PRJ_with_OBS to produce a DF with the PRJ data.
    Could have been included in utils.py based on dependencies, but utils_imod.py fits it better as it's almost alwaysused after read_PRJ_with_OBS (so the libs will be already loaded)."""
    # print = print if verbose else lambda *a, **k: None
    
    d_Pa = get_MdlN_paths(MdlN)

    Mdl = ''.join([c for c in MdlN if not c.isdigit()])
    Pa_AppData = os.path.normpath(PJ(os.getenv('APPDATA'), '../'))
    t_Pa_replace = (Pa_AppData, PJ(Pa_WS, 'models', Mdl)) # For some reason imod.idf.read reads the path incorrectly, so I have to replace the incorrect part.

    d_PRJ, OBS = read_PRJ_with_OBS(d_Pa['PRJ'])

    columns = ['package', 'parameter','time', 'active', 'is_constant', 'layer', 'factor', 'addition', 'constant', 'path']
    DF = pd.DataFrame(columns=columns) # Main DF to store all the packages

    vprint(f' --- Reading PRJ Packages into DF ---')
    for Pkg_name in list(d_PRJ.keys()): # Iterate over packages
        vprint(f"\t{Pkg_name:<7}\t...\t", end='')
        try:
            Pkg = d_PRJ[Pkg_name]

            if int(Pkg['active']): # if the package is active, process it
                l_Par = [k for k in Pkg.keys() if k not in {'active', 'n_system', 'time'}] # Make list from package keys/parameters
                for Par in l_Par[:]: # Iterate over parameters

                    for N, L in enumerate(Pkg[Par]): # differentiate between packages (have time) and modules.
                        Ln_DF_path = {**L, "package": Pkg_name, "parameter": Par} #, 'Pa_type':L['path'].suffix.lower()} #, "metadata": L}
                        
                        if ('time' in d_PRJ[Pkg_name].keys()):
                            if (Pkg['n_system'] > 1):
                                DF.loc[f'{Pkg_name.upper()}_{Par}_Sys{(N)%Pkg['n_system']+1}_{L['time']}'] = Ln_DF_path
                            elif (Pkg['n_system']==1):
                                DF.loc[f"{Pkg_name.upper()}_{Par}"] = Ln_DF_path
                        else:
                            if (Pkg['n_system'] > 1):
                                DF.loc[f'{Pkg_name.upper()}_{Par}_Sys{(N)%Pkg['n_system']+1}'] = Ln_DF_path
                            elif (Pkg['n_system']==1):
                                DF.loc[f"{Pkg_name.upper()}_{Par}"] = Ln_DF_path
                vprint('🟢')
            else:
                vprint(f'\u2012 the package is innactive.')
        except:
            DF.loc[f'{Pkg_name.upper()}'] = "-"
            DF.loc[f'{Pkg_name.upper()}', 'active'] = 'Failed to read package'
            vprint('🟡')
    vprint('🟢🟢🟢')
    vprint(f' {"-"*100}')

    DF['package'] = DF['package'].str.replace('(',"").str.replace(')','').str.upper()
    DF['suffix'] = DF['path'].apply(lambda x: x.suffix.lower() if hasattr(x, 'suffix') else "-")  # Check if 'suffix' exists # Make suffix column so that paths can be categorized
    DF['path'] = DF['path'].astype('string') # Convert path to string so that the wrong part of the path can be .replace()ed
    DF['MdlN'] = DF['path'].str.split("_").str[-1].str.split('.').str[0]
    DF['path'] = DF['path'].str.replace(*t_Pa_replace, regex=False) # Replace incorrect part of paths. I'm not sure why iMOD doesn't read them right. Maybe cause they're relative it's assumed they start form a directory which is incorrect.
    DF = DF.loc[:, list(DF.columns[:2]) + ['MdlN'] + list(DF.columns[2:-3]) + ['suffix', DF.columns[-3]]] # Rearrange columns

    return DF

def open_PRJ_with_OBS(Pa_PRJ): #666 gives error cause it tries to read files referenced by PRJ using a default user directory as base. 
    """imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines."""
    with open(Pa_PRJ, "r") as f:
        lines = f.readlines()

    l_filtered_Lns, l_OBS_Lns = [], []
    skip_block = False

    for line in lines:
        if "(obs)" in line.lower():  # Start of OBS block
            skip_block = True
            l_OBS_Lns.append(line)  # Keep the header
        elif skip_block and line.strip() == "":  # End of OBS block
            skip_block = False
        elif skip_block:
            l_OBS_Lns.append(line)  # Store OBS content
        else:
            l_filtered_Lns.append(line)  # Keep everything else

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".prj") as temp_file:
        temp_file.writelines(l_filtered_Lns)
        Pa_PRJ_temp = temp_file.name

    PRJ = imod.formats.prj.open_projectfile_data(Pa_PRJ_temp) # Load the PRJ file without OBS
    os.remove(Pa_PRJ_temp) # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns
#-----------------------------------------------------------------------------------------------------------------------------------

# PrSimP related -------------------------------------------------------------------------------------------------------------------
def add_OBS(MdlN:str, Opt:str="BEGIN OPTIONS\nEND OPTIONS"):
    """Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'"""

    vprint(Pre_Sign)
    vprint('Running add_OBS ...')
    d_Pa = get_MdlN_paths(MdlN) # Get default directories
    Pa_MdlN, Pa_INI, Pa_PRJ = (d_Pa[k] for k in ['Pa_MdlN', 'INI', 'PRJ']) # and pass them to objects that will be used in the function
    
    # Extract info from INI file.
    d_INI = INI_to_d(Pa_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    # N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ), 

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = read_PRJ_with_OBS(Pa_PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)" # Regex pattern to extract file paths ending in .ipf
    l_IPF = [match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)] # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        Pa_OBS_IPF = os.path.abspath(PJ(Pa_MdlN, path)) # path of IPF file. To be read.
        OBS_IPF_Fi = PBN(Pa_OBS_IPF) # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}.OBS6') #path of OBS file. To be written.
        else:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}_N{i}.OBS6') #path of OBS file. To be written.

        DF_OBS_IPF = read_IPF_Spa(Pa_OBS_IPF) # Get list of OBS items (without temporal dimension, as it's uneccessary for the OBS file, and takes ages to load)
        DF_OBS_IPF_MdlAa = DF_OBS_IPF.loc[ ( (DF_OBS_IPF['X']>Xmin) & (DF_OBS_IPF['X']<Xmax ) ) &
                                           ( (DF_OBS_IPF['Y']>Ymin) & (DF_OBS_IPF['Y']<Ymax ) )].copy() # Slice to OBS within the Mdl Aa (using INI window)
        
        DF_OBS_IPF_MdlAa['C'] = ( (DF_OBS_IPF_MdlAa['X']-Xmin) / cellsize ).astype(np.int32) + 1 # Calculate Cs. Xmin at the origin of the model.
        DF_OBS_IPF_MdlAa['R'] = ( -(DF_OBS_IPF_MdlAa['Y']-Ymax) / cellsize ).astype(np.int32) + 1 # Calculate Rs. Ymax at the origin of the model.

        DF_OBS_IPF_MdlAa.sort_values(by=["L", "R", "C"], ascending=[True, True, True], inplace=True) # Let's sort the DF by L, R, C

        with open(Pa_OBS, "w") as f: # write OBS file(s)
            #vprint(Pa_MdlN, path, Pa_OBS_IPF, sep='\n')
            f.write(f"# created from {Pa_OBS_IPF}\n")
            f.write(Opt.encode().decode('unicode_escape')) # write optional block
            f.write(f"\n\nBEGIN CONTINUOUS FILEOUT OBS_{OBS_IPF_Fi.split('.')[0]}.csv\n")
            
            for _, row in DF_OBS_IPF_MdlAa.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f" {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n")
            
            f.write("END CONTINUOUS\n")
    
        # Open NAM file and add OBS file to it
        lock = FL(d_Pa['NAM_Mdl'] + '.lock')  # Create a file lock to prevent concurrent writes
        with lock, open(d_Pa['NAM_Mdl'], 'r+') as f:

            l_NAM = f.read().split('END PACKAGES')
            f.seek(0); f.truncate()                  # overwrite in-place
            Pa_OBS_Rel = os.path.relpath(Pa_OBS, Pa_MdlN)

            f.write(l_NAM[0])
            f.write(fr' OBS6 .\{Pa_OBS_Rel} OBS_{OBS_IPF_Fi.split(".")[0]}')
            f.write('\nEND PACKAGES')

            f.flush(); os.fsync(f.fileno())          # ensure it’s on disk
            # lock is released automatically when the with-block closes
        vprint(f'🟢 - {Pa_OBS} has been added successfully!')
    vprint(Sign)
#-----------------------------------------------------------------------------------------------------------------------------------

# run_Mdl --------------------------------------------------------------------------------------------------------------------------
def run_Mdl(Se_Ln, DF_Opt): #666 think if this can be improved to take only 1 argument. Function becomes redundant from v.1.0.0, as snakemae files are used instead.
    """Runs the model from PrP, to PrSimP, to PP.
    Requires:
    - `Se_Ln`: A Pandas Series (row from the RunLog sheet of RunLog.xlsx)
    - `DF_Opt`: A Pandas DataFrame (PP_Opt sheet of the same spreadsheet)
    """
    MdlN = Se_Ln['MdlN']
    vprint(f"--- Executing RunMng for {MdlN}")

    # Get default directories
    d_Pa = get_MdlN_paths(MdlN)
    MdlN_B, Pa_Mdl, Pa_MdlN, Pa_INI, Pa_BAT, Pa_PRJ = (d_Pa[k] for k in ['MdlN_B', 'Pa_Mdl', 'Pa_MdlN', 'INI', 'BAT', 'PRJ'])
        
    # Define commands and their working directories
    d_Cmds = {Pa_BAT: PDN(Pa_BAT),
              "activate WS": PDN(Pa_BAT),
              f"WS_Mdl add_OBS {MdlN} {DF_Opt.loc[DF_Opt['MdlN']==MdlN, 'add_OBS'].values[0]}": PDN(Pa_BAT),
              r'.\RUN.BAT': Pa_MdlN}

    # Generate log file path
    log_file = PJ(Pa_MdlN, f'Tmnl_Out_{MdlN}.txt')
    MDs(PDN(log_file), exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as f:
        for Cmd, cwd in d_Cmds.items():
            try:
                vprint(f" -- Executing: {Cmd}")

                # Run command in blocking mode, capturing output live
                process = sp.run(Cmd, shell=True, cwd=cwd,
                                 capture_output=True, text=True, check=True)

                # Write output to file
                f.write(f'{"*"*50}  START OF COMMAND   {"*"*50}\n')
                f.write(f"Command: {Cmd}\n{"-"*120}\n\n")
                f.write(f"Output:\n{process.stdout}\n{"-"*120}\n\n")
                f.write(f"Errors:\n{process.stderr}\n{"-"*120}\n\n")
                f.write(f"Return Code: {process.returncode}\n{"-"*120}\n\n")
                f.write(f'{"*"*50}  END OF COMMAND     {"*"*50}\n\n')

                vprint("  - 🟢")

            except sp.CalledProcessError as e:
                print(f"  - 🔴: {Cmd}\nError: {e.stderr}")
                f.write(f"ERROR: {e.stderr}\n")

def run_Mdl_print_only(Se_Ln, DF_Opt): #666 think if this can be improved to take only 1 argument.
    """Runs the model from PrP, to PrSimP, to PP.
    Requires:
    - `Se_Ln`: A Pandas Series (row from the RunLog sheet of RunLog.xlsx)
    - `DF_Opt`: A Pandas DataFrame (PP_Opt sheet of the same spreadsheet)
    """
    MdlN = Se_Ln['MdlN']
    vprint(f"--- Executing RunMng for {MdlN}")

    # Get default directories
    d_Pa = get_MdlN_paths(MdlN)
    MdlN_B, Pa_Mdl, Pa_MdlN, Pa_INI, Pa_BAT, Pa_PRJ = (d_Pa[k] for k in ['MdlN_B', 'Pa_Mdl', 'Pa_MdlN', 'INI', 'BAT', 'PRJ'])
        
    # Define commands and their working directories
    d_Cmds = {Pa_BAT: PDN(Pa_BAT),
              "activate WS": PDN(Pa_BAT),
              f"WS_Mdl add_OBS {MdlN} {DF_Opt.loc[DF_Opt['MdlN']==MdlN, 'add_OBS'].values[0]}": PDN(Pa_BAT),
              r'.\RUN.BAT': Pa_MdlN}

    # Generate log file path
    log_file = PJ(Pa_MdlN, f'Tmnl_Out_{MdlN}.txt')
    MDs(PDN(log_file), exist_ok=True)

    for Cmd, cwd in d_Cmds.items():
        try:
            vprint(f" -- Executing: {Cmd}\nin {cwd}")
            vprint("  - 🟢")

        except sp.CalledProcessError as e:
            print(f"  - 🔴: {Cmd}\nError: {e.stderr}")

def run_Mdl_parallel(DF, DF_Opt):
    queued_Sims = DF.loc[DF['Status'] == 'Queued']
    num_cores = min(len(queued_Sims), cpu_count())
    processes = []

    for _, Se_Ln in queued_Sims.iterrows():
        p = Process(target=run_Mdl, args=(Se_Ln, DF_Opt))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
#-----------------------------------------------------------------------------------------------------------------------------------

# IDF processing -------------------------------------------------------------------------------------------------------------------
def IDFs_to_DF(S_Pa_IDF):
    """ Reads all .IDF Fis listed in a S_Fi_IDF into DF['IDF']. Returns the DF containing Fi_names and the IDF contents.
        Pa_Fo is the path of the Fo where th files are stored in."""

    DF = pd.DataFrame({'path': S_Pa_IDF, 'IDF': None})

    for i, p in tqdm(DF['path'].items(), desc="Loading .IDF files", total=len(DF['path'])):
        if p.endswith('.IDF'):  # Ensure only .IDF files are processed
            try:    # Read the .IDF file into an xA DataA
                DF.at[i, 'IDF'] = imod.idf.read( p )
            except Exception as e:
                print(f"Error reading {p}: {e}")
    return DF
#-----------------------------------------------------------------------------------------------------------------------------------
