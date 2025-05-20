# ***** Similar to utils.py, but those utilize imod, which takes a long time to load. *****
import os
import tempfile
import imod
from .utils import Sign, Pre_Sign, read_IPF_Spa, INI_to_d, get_MdlN_paths
import numpy as np
import subprocess as sp
from multiprocessing import Process, cpu_count
import re
import pandas as pd
from tqdm import tqdm # Track progress of the loop

# PRJ related ----------------------------------------------------------------------------------------------------------------------
def read_PRJ_with_OBS(path_PRJ):
    """imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines."""
    with open(path_PRJ, "r") as f:
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
        path_PRJ_temp = temp_file.name

    PRJ = imod.formats.prj.read_projectfile(path_PRJ_temp) # Load the PRJ file without OBS
    os.remove(path_PRJ_temp) # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns

def open_PRJ_with_OBS(path_PRJ): #666 gives error cause it tries to read files referenced by PRJ using a default user directory as base. 
    """imod.formats.prj.read_projectfile struggles with .prj files that contain OBS blocks. This will read the PRJ file and return a tuple. The first item is a PRJ dictionary (as imod.formats.prj would return) and also a list of the OBS block lines."""
    with open(path_PRJ, "r") as f:
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
        path_PRJ_temp = temp_file.name

    PRJ = imod.formats.prj.open_projectfile_data(path_PRJ_temp) # Load the PRJ file without OBS
    os.remove(path_PRJ_temp) # Delete temp PRJ file as it's not needed anymore.

    return PRJ, l_OBS_Lns
#-----------------------------------------------------------------------------------------------------------------------------------

# PrSimP related -------------------------------------------------------------------------------------------------------------------
def add_OBS(MdlN:str, Opt:str="BEGIN OPTIONS\nEND OPTIONS"):
    """Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'"""

    print(Pre_Sign)
    print('Running add_OBS ...')
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    path_MdlN, path_INI, path_PRJ = (d_paths[k] for k in ['path_MdlN', "path_INI_S", "path_PRJ_S"]) # and pass them to objects that will be used in the function
    
    # Extract info from INI file.
    d_INI = INI_to_d(path_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    # N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ), 

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = read_PRJ_with_OBS(path_PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)" # Regex pattern to extract file paths ending in .ipf
    l_IPF = [match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)] # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        path_OBS_IPF = os.path.abspath(os.path.join(path_MdlN, path)) # path of IPF file. To be read.
        OBS_IPF_Fi = os.path.basename(path_OBS_IPF) # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            path_OBS = os.path.join(path_MdlN, f'GWF_1/MODELINPUT/{MdlN}.OBS') #path of OBS file. To be written.
        else:
            path_OBS = os.path.join(path_MdlN, f'GWF_1/MODELINPUT/{MdlN}_N{i}.OBS') #path of OBS file. To be written.

        DF_OBS_IPF = read_IPF_Spa(path_OBS_IPF) # Get list of OBS items (without temporal dimension, as it's uneccessary for the OBS file, and takes ages to load)
        DF_OBS_IPF_MdlAa = DF_OBS_IPF.loc[ ( (DF_OBS_IPF['X']>Xmin) & (DF_OBS_IPF['X']<Xmax ) ) &
                                           ( (DF_OBS_IPF['Y']>Ymin) & (DF_OBS_IPF['Y']<Ymax ) )].copy() # Slice to OBS within the Mdl Aa (using INI window)
        
        DF_OBS_IPF_MdlAa['C'] = ( (DF_OBS_IPF_MdlAa['X']-Xmin) / cellsize ).astype(np.int32) + 1 # Calculate Cs. Xmin at the origin of the model.
        DF_OBS_IPF_MdlAa['R'] = ( -(DF_OBS_IPF_MdlAa['Y']-Ymax) / cellsize ).astype(np.int32) + 1 # Calculate Rs. Ymax at the origin of the model.

        DF_OBS_IPF_MdlAa.sort_values(by=["L", "R", "C"], ascending=[True, True, True], inplace=True) # Let's sort the DF by L, R, C

        with open(path_OBS, "w") as f: # write OBS file(s)
            print(path_MdlN, path, path_OBS_IPF, sep='\n')
            f.write(f"# created from {path_OBS_IPF}\n")
            f.write(Opt.encode().decode('unicode_escape')) # write optional block
            f.write(f"\n\nBEGIN CONTINUOUS FILEOUT OBS_{OBS_IPF_Fi.split('.')[0]}.csv\n")
            
            for _, row in DF_OBS_IPF_MdlAa.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f" {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n")
            
            f.write("END CONTINUOUS\n")
    
        # Open NAM file and add OBS file to it
        with open(os.path.join(path_MdlN, 'GWF_1', MdlN+'.nam'), 'r') as f1:
            NAM = f1.read()
        l_NAM = NAM.split('END PACKAGES')
        with open(os.path.join(path_MdlN, 'GWF_1', MdlN+'.nam'), 'w') as f2:
            path_OBS_Rel = os.path.relpath(path_OBS, path_MdlN) # Creates Rel path from full path.
            f2.write(l_NAM[0])
            f2.write(fr' OBS6 .\{path_OBS_Rel} OBS_{OBS_IPF_Fi.split('.')[0]}')
            f2.write('\nEND PACKAGES')
        print(f'{path_OBS} has been added successfully!')
    print(Sign)
#-----------------------------------------------------------------------------------------------------------------------------------

# run_Mdl --------------------------------------------------------------------------------------------------------------------------
def run_Mdl(Se_Ln, DF_Opt): #666 think if this can be improved to take only 1 argument. Function becomes redundant from v.1.0.0, as snakemae files are used instead.
    """Runs the model from PrP, to PrSimP, to PP.
    Requires:
    - `Se_Ln`: A Pandas Series (row from the RunLog sheet of RunLog.xlsx)
    - `DF_Opt`: A Pandas DataFrame (PP_Opt sheet of the same spreadsheet)
    """
    MdlN = Se_Ln['MdlN']
    print(f"--- Executing RunMng for {MdlN}")

    # Get default directories
    d_paths = get_MdlN_paths(MdlN)
    MdlN_B, path_Mdl, path_MdlN, path_INI, path_BAT, path_PRJ = (d_paths[k] for k in ['MdlN_B', 'path_Mdl', 'path_MdlN', "path_INI_S", "path_BAT_S", "path_PRJ_S"])
        
    # Define commands and their working directories
    d_Cmds = {path_BAT: os.path.dirname(path_BAT),
              "activate WS": os.path.dirname(path_BAT),
              f"WS_Mdl add_OBS {MdlN} {DF_Opt.loc[DF_Opt['MdlN']==MdlN, 'add_OBS'].values[0]}": os.path.dirname(path_BAT),
              r'.\RUN.BAT': path_MdlN}

    # Generate log file path
    log_file = os.path.join(path_MdlN, f'Tmnl_Out_{MdlN}.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as f:
        for Cmd, cwd in d_Cmds.items():
            try:
                print(f" -- Executing: {Cmd}")

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

                print(f"  - ✓")

            except sp.CalledProcessError as e:
                print(f"  - ❌: {Cmd}\nError: {e.stderr}")
                f.write(f"ERROR: {e.stderr}\n")

def run_Mdl_print_only(Se_Ln, DF_Opt): #666 think if this can be improved to take only 1 argument.
    """Runs the model from PrP, to PrSimP, to PP.
    Requires:
    - `Se_Ln`: A Pandas Series (row from the RunLog sheet of RunLog.xlsx)
    - `DF_Opt`: A Pandas DataFrame (PP_Opt sheet of the same spreadsheet)
    """
    MdlN = Se_Ln['MdlN']
    print(f"--- Executing RunMng for {MdlN}")

    # Get default directories
    d_paths = get_MdlN_paths(MdlN)
    MdlN_B, path_Mdl, path_MdlN, path_INI, path_BAT, path_PRJ = (d_paths[k] for k in ['MdlN_B', 'path_Mdl', 'path_MdlN', "path_INI_S", "path_BAT_S", "path_PRJ_S"])
        
    # Define commands and their working directories
    d_Cmds = {path_BAT: os.path.dirname(path_BAT),
              "activate WS": os.path.dirname(path_BAT),
              f"WS_Mdl add_OBS {MdlN} {DF_Opt.loc[DF_Opt['MdlN']==MdlN, 'add_OBS'].values[0]}": os.path.dirname(path_BAT),
              r'.\RUN.BAT': path_MdlN}

    # Generate log file path
    log_file = os.path.join(path_MdlN, f'Tmnl_Out_{MdlN}.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    for Cmd, cwd in d_Cmds.items():
        try:
            print(f" -- Executing: {Cmd}\nin {cwd}")
            print(f"  - ✓")

        except sp.CalledProcessError as e:
            print(f"  - ❌: {Cmd}\nError: {e.stderr}")

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
def IDFs_to_DF(S_path_IDF):
    """ Reads all .IDF Fis listed in a S_Fi_IDF into DF['IDF']. Returns the DF containing Fi_names and the IDF contents.
        path_Fo is the path of the Fo where th files are stored in."""

    DF = pd.DataFrame({'path': S_path_IDF, 'IDF': None})

    for i, p in tqdm(DF['path'].items(), desc="Loading .IDF files", total=len(DF['path'])):
        if p.endswith('.IDF'):  # Ensure only .IDF files are processed
            try:    # Read the .IDF file into an xA DataA
                DF.at[i, 'IDF'] = imod.idf.read( p )
            except Exception as e:
                print(f"Error reading {p}: {e}")
    return DF
#-----------------------------------------------------------------------------------------------------------------------------------