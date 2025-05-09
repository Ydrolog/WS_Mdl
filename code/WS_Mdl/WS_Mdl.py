import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import imod
from datetime import datetime as DT
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict
import shutil as sh
import warnings
import tempfile
import re
import subprocess as sp
from multiprocessing import Process, cpu_count
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import time

## How to import:
# import WS_Mdl as WS
# As long as it is installed (check ../../READ_ME.md for instructions), this should work

path_WS = 'C:/OD/WS_Mdl'
crs = "EPSG:28992"
path_RunLog = os.path.join(path_WS, 'Mng/WS_RunLog.xlsx')
path_log = os.path.join(path_WS, 'Mng/log.csv')
Sign = f"\nGoodbye, friend.\n{'*'*80}\n"

## Can make get paths function that will provide the general directories, like path_WS, path_Mdl. Those can be derived from a folder structure.

# Get paths ---------------------------------------------------------------------------------
def MdlN_Se_from_RunLog(MdlN):
    """Returns RunLog line that corresponds to MdlN as a S."""
    DF = pd.read_excel(os.path.join(path_WS, 'Mng/WS_RunLog.xlsx'), sheet_name='RunLog')    
    S = DF.loc[DF['MdlN']==MdlN].squeeze()
    return S

def paths_from_MdlN_Se(S, MdlN_S):
    """Takes in S, returns relevant paths."""
    Mdl, SimN_B = S[['model alias', 'B SimN']]
    MdlN_B = Mdl + str(SimN_B)

    path_Mdl = os.path.join(path_WS, f'models/{Mdl}')
    path_MdlN = os.path.join(path_Mdl, f"Sim/{MdlN_S}")
    path_INI_B = os.path.join(path_Mdl, f'code/Mdl_Prep/Mdl_Prep_{MdlN_B}.ini')
    path_BAT_B = os.path.join(path_Mdl, f'code/Mdl_Prep/Mdl_Prep_{MdlN_B}.bat')
    path_PRJ_B = os.path.join(path_Mdl, f'In/PRJ/{MdlN_B}.prj')
    path_Smk_B = os.path.join(path_Mdl, f'code/snakemake/{MdlN_B}.smk')
    path_INI_S = path_INI_B.replace(MdlN_B, MdlN_S)
    path_BAT_S = path_BAT_B.replace(MdlN_B, MdlN_S)
    path_PRJ_S = path_PRJ_B.replace(MdlN_B, MdlN_S)
    path_Smk_S = path_Smk_B.replace(MdlN_B, MdlN_S)
    
    return {'MdlN_B': MdlN_B,
            'path_Mdl': path_Mdl,
            'path_MdlN': path_MdlN,
            'path_INI_B': path_INI_B,
            'path_INI_S': path_INI_S,
            'path_BAT_B': path_BAT_B,
            'path_BAT_S': path_BAT_S,
            'path_PRJ_B': path_PRJ_B,
            'path_PRJ_S': path_PRJ_S,
            'path_Smk_B': path_Smk_B,
            'path_Smk_S': path_Smk_S}

def get_MdlN_paths(MdlN_S: str): #666 Can be split into two as both S and B aren't allways needed. Or better, I can make a new function that does that for just 1 run.
    """ Returns a dictionary of useful object (MdlN_B, directories etc.) for a given model. Those need to then be passed to arguments, e.g. path_INI_B = Dft_paths['path_INI_N']."""
    return paths_from_MdlN_Se( MdlN_Se_from_RunLog((MdlN_S)), MdlN_S )
# ---------------------------------------------------------------------------------


# READ FILES ---------------------------------------------------------------------------------
def read_IPF_Spa(path_IPF):
    """Reads IPF file without temporal component - i.e. no linked TS text files. Returns a DF created from just the IPF file.""" 
    with open(path_IPF, "r") as f:
        l_Ln = f.readlines()

    N_C = int(l_Ln[1].strip())  # Number of columns
    l_C_Nm = [l_Ln[I + 2].split("\n")[0] for I in range(N_C)] # Extract column names
    DF_IPF = pd.read_csv(path_IPF, skiprows=2+N_C+1, names=l_C_Nm)

    return DF_IPF

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

def open_PRJ_with_OBS(path_PRJ):
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
    return d_INI

def Mdl_Dmns_from_INI(path_INI): # 666 Can be improved. It should take a MdlN instead of a path. Makes things easier.
    """Returns model dimension parameters. Common use:
    Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C = WS.Mdl_Dmns_from_INI(path)"""
    d_INI = INI_to_d(path_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ), 
    return Xmin, Ymin, Xmax, Ymax, cellsize, N_R, N_C
#---------------------------------------------------------------------------------

# Convert Files ---------------------------------------------------------------------------------
def IDF_to_TIF(path_IDF: str, path_TIF: Optional[str] = None, MtDt: Optional[Dict] = None, crs=crs):
    """ Converts IDF file to TIF file.
        If path_TIF is not provided, it'll be the same as path_IDF, except for the file type ending.
        crs (coordinate reference system) is set to the Amerfoot crs by default, but can be changed for other projects."""
    try:
        A, MtDt = imod.idf.read(path_IDF)

        Ogn_DT = DT.fromtimestamp(os.path.getctime(path_IDF)).strftime('%Y-%m-%d %H:%M:%S') # Get OG (IDF) file's date modified.
        Cvt_DT = DT.now().strftime('%Y-%m-%d %H:%M:%S') # Get current time, to write time of convertion to comment

        N_R, N_C = A.shape

        transform = from_bounds(west=MtDt['xmin'],
                                south=MtDt['ymin'],
                                east=MtDt['xmax'],
                                north=MtDt['ymax'],
                                width=N_C,
                                height=N_R)
        meta = {"driver": "GTiff",
                "height": N_R,
                "width": N_C,
                "count": 1,
                "dtype": str(A.dtype),
                "crs": crs,
                "transform": transform}

        if not path_TIF:
            path_TIF = os.path.splitext(path_IDF)[0] + '.tif'

        with rasterio.open(path_TIF, "w", **meta) as Dst:
            Dst.write(A, 1)  # Write band 1

            Cvt_MtDt = {'COMMENT':(f"Converted from IDF on {Cvt_DT}."
                                f"Original file created on {Ogn_DT}."
                                f"Original IDF file location: {path_IDF}")}
                    
            if MtDt: # If project metadata exists, store it separately
                project_metadata = {f"USER_{k}": str(v) for k, v in MtDt.items()}
                Cvt_MtDt.update(project_metadata)

            Dst.update_tags(**Cvt_MtDt)
        print(f"\u2713 {path_TIF} has been saved (GeoTIFF) with conversion and project metadata.")
    except Exception as e:
        print(f"\u274C \n{e}")
    print(Sign)

# def l_IDF_to_TIF(l_IDF, Dir_Out):
#     """#666 under construction. The aim of this is to make a multi-band tif file instead of multiple single-band tif files, for each parameter."""
#     DA = imod.formats.idf.open(l_IDF)#.sel(x=slice(Xmin, Xmax), y=slice(Ymax, Ymin)) # Read IDF files to am xarray.DataArray and slice it to model area (read from INI file)
#     DA = DA.rio.write_crs(crs)  # Set Dutch RD New projection
#     DA.rio.to_raster(Dir_Out)    
#---------------------------------------------------------------------------------

class Vld_Mtc:
    formulas = {
        "NSE": lambda obs, sim: 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
        "RMSE": lambda obs, sim: np.sqrt(mean_squared_error(obs, sim)),
        "MAE": lambda obs, sim: np.mean(np.abs(obs - sim)),
        "Correlation": lambda obs, sim: np.corrcoef(obs, sim)[0, 1],  # Pearson correlation coefficient
        "Bias Ratio": lambda obs, sim: np.mean(sim) / np.mean(obs),  # β = mean(sim) / mean(obs)
        "Variability Ratio": lambda obs, sim: (np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)),  # γ'
        "KGE": lambda obs, sim: 1 - np.sqrt(
            (np.corrcoef(obs, sim)[0, 1] - 1) ** 2 + 
            (np.mean(sim) / np.mean(obs) - 1) ** 2 + 
            ((np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)) - 1) ** 2
        )  # Kling-Gupta Efficiency (KGE')
    }

    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        self.formula = self.formulas.get(name)

    def compute(self, obs, sim):
        if self.formula:
            return self.formula(obs, sim)
        else:
            raise ValueError(f"Formula for {self.name} not found!")
        
# Mdl PP ---------------------------------------------------------------------------------
def S_from_B(MdlN:str):
    """Copies files that contain Sim options from the B Sim, renames them for the S Sim, and opens them in the default file editor. Assumes default WS_Mdl folder structure (as described in READ_ME.MD)."""
    
    print('-'*100)
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    MdlN_B, path_INI_B, path_INI_S, path_BAT_B, path_BAT_S, path_Smk_S, path_Smk_B, path_PRJ_B, path_PRJ_S = (d_paths[k] for k in ['MdlN_B', "path_INI_B", "path_INI_S", "path_BAT_B", "path_BAT_S", "path_Smk_S", "path_Smk_B", "path_PRJ_B", "path_PRJ_S"]) # and pass them to objects that will be used in the function

    # Copy .INI, .bat, .prj and make default (those apply to every Sim) modifications
    for path_B, path_S in zip([path_Smk_B, path_BAT_B, path_INI_B], [path_Smk_S, path_BAT_S, path_INI_S]):
        print(path_B)
        print(path_S)
        if not os.path.exists(path_S): # Replace the MdlN of with the new one, so that we don't have to do it manually.
            sh.copy2(path_B, path_S)
            with open(path_S, 'r') as f1:
                contents = f1.read()
            with open(path_S, 'w') as f2:
                f2.write(contents.replace(MdlN_B, MdlN))
            if ".bat" not in path_B.lower():
                os.startfile(path_S) # Then we'll open it to make any other changes we want to make. Except if it's the BAT file
            print(f'\u2713 - {path_S.split('/')[-1]} created successfully! (from {path_B})')
        else:
            print(f"\u274C - {path_S.split('/')[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.")

    if not os.path.exists(path_PRJ_S): # For the PRJ file, there is no default replacement, so we'll just copy.
        sh.copy2(path_PRJ_B, path_PRJ_S)
        os.startfile(path_PRJ_S) # Then we'll open it to make any other changes we want to make.
        print(f'\u2713 - {path_PRJ_S.split('/')[-1]} created successfully! (from {path_PRJ_B})')        
    else:
        print(f"\u274C - {path_PRJ_S.split('/')[-1]} already exists. If you want it to be replaced, you have to delete it manually before running this command.")
    print(Sign)

def S_from_B_undo(MdlN:str):
    """Will undo S_from_B by deletting S files"""
    print('*'*80)
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    MdlN_B, path_INI_B, path_INI_S, path_BAT_B, path_BAT_S, path_Smk_S, path_Smk_B, path_PRJ_B, path_PRJ_S = (d_paths[k] for k in ['MdlN_B', "path_INI_B", "path_INI_S", "path_BAT_B", "path_BAT_S", "path_Smk_S", "path_Smk_B", "path_PRJ_B", "path_PRJ_S"]) # and pass them to objects that will be used in the function

    confirm = input(f"Are you sure you want to delete the Cfg files (.smk, .ini, .bat, .prj) for {MdlN}? (y/n): ").strip().lower()
    if confirm == 'y':
        for path_S in [path_Smk_S, path_BAT_S, path_INI_S, path_PRJ_S]:
            os.remove(path_S) # Delete the S files
            print(f'\u2713 - {path_S.split("/")[-1]} deleted successfully!')
    print(Sign)

def add_OBS(MdlN:str, Opt:str="BEGIN OPTIONS\nEND OPTIONS"):
    """Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'"""

    print('-'*80)
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
    print(' finished successfully!')
    print('-'*80)

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

def Up_log(MdlN: str,
           d_Up: dict,
           path_log=os.path.join(path_WS, 'Mng/log.csv')):
    """Update log.csv based on MdlN and key of `updates`."""
    DF = pd.read_csv(path_log, index_col=0)  # Assumes log.csv exists.

    for key, value in d_Up.items():  # Update the relevant cells
        DF.at[MdlN, key] = value

    while True: # Wait for file to be closed if it's open
        try:
            DF.to_csv(path_log, date_format='%Y-%m-%d %H:%M')  # Save back to CSV
            break  # Break if successful
        except PermissionError:
            input("log.csv is open. Press Enter after closing the file...")  # Wait for user input

def DA_to_TIF(DA, path_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 2D xarray.DataArray (shape = [y, x]) to a single-band GeoTIFF.
    - DA: 2D xarray.DataArray with shape [y, x]
    - path_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata for this single band.
      Must contain exactly 1 item: {band_description: band_metadata_dict}
    - crs: Coordinate Reference System (optional)."""    
    
    if len(d_MtDt) != 1: # We expect exactly one band, so parse the single (key, value) from d_MtDt
        raise ValueError("DA_to_TIF expects exactly 1 item in d_MtDt for a 2D DataArray.")

    (band_key, band_meta) = list(d_MtDt.items())[0]

    transform = DA.rio.transform() # Build transform from DA

    with rasterio.open(path_Out,
                       "w",
                       driver="GTiff",
                       height=DA.shape[0],
                       width=DA.shape[1],
                       count=1,                   # single band
                       dtype=str(DA.dtype),
                       crs=crs,
                       transform=transform) as Dst:
        Dst.write(DA.values, 1) # Write the 2D data as band 1
        Dst.set_band_description(1, band_key) # Give the band a useful name
        Dst.update_tags(1, **band_meta) # Write each row field as a separate metadata tag on this band
    if _print:
        print(f"DA_to_TIF finished successfully for: {path_Out}")

def DA_to_MBTIF(DA, path_Out, d_MtDt, crs=crs, _print=False):
    """ Write a 3D xarray.DataArray (shape = [n_bands, y, x]) to a GeoTIFF. This bypasses rioxarray.to_raster() entirely, letting us set per-band descriptions and metadata in a single pass.
    - DA: 3D xarray.DataArray with shape [n_bands, y, x]
    - path_Out: Path to the output GeoTIFF file.
    - d_MtDt: Dictionary with metadata to be written to the GeoTIFF file. Each key is a band index (1-based) and each value is a dictionary of metadata tags.
    - crs: Coordinate Reference System (optional)."""

    band_keys, band_MtDt = zip(*d_MtDt.items())

    transform = DA.rio.transform()

    with rasterio.open(path_Out, #666 add ask-to-overwrite function (preferably to any function/command in this Lib that writes a file.)
                       "w",
                       driver="GTiff",
                       height=DA.shape[1],
                       width=DA.shape[2],
                       count=DA.shape[0],
                       dtype=str(DA.dtype),
                       crs=crs,
                       transform=transform,
                       photometric="MINISBLACK") as Dst:
        for i in range(DA.shape[0]): # Write each band.
            Dst.write(DA[i].values, i + 1) # Write the actual pixels for this band (i+1 is the band index in Rasterio)
            Dst.set_band_description(i + 1, band_keys[i]) # Set a band description that QGIS will show as "Band 01: <description>"
            Dst.update_tags(i + 1, **band_MtDt[i]) # Write each row field as a separate metadata tag on this band

        if "all" in d_MtDt: # If "all" exists, write dataset-wide metadata (NOT tied to a band)
            Dst.update_tags(**d_MtDt["all"])  # Set global metadata for the whole dataset
            
    if _print:
        print(f"DA_to_MBTIF finished successfully for: {path_Out}")
# ---------------------------------------------------------------------------------

def RunMng(cores=None, DAG:bool=True):
    """Read the RunLog, and for each queued model, run the corresponding Snakemake file."""
    if cores is None:
        cores = max(cpu_count() - 2, 1) # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    print(f"\n{'*'*80}\nRunMng will run all Sims that are queued in the RunLog.\n")

    print(f"--- Reading RunLog ...", end='')
    DF = pd.read_excel(path_RunLog, sheet_name='RunLog').dropna(subset='runN') # Read RunLog
    DF_q = DF.loc[ (DF['Start Status'] == 'Queued') & ((DF['End Status'].isna()) | (DF['End Status']=='Failed')) ] # _q for queued. Only Run Queued runs that aren't running or have finished.
    print(' completed!\n')

    print('--- Running snakemake files:')
    if DF_q.empty:
        print("❌ - No queued runs found in the RunLog.")
    else:
        for i, Se_Ln in DF_q.iterrows():
            path_Smk = os.path.join(path_WS, f'models/{Se_Ln["model alias"]}/code/snakemake/{Se_Ln["MdlN"]}.smk')
            path_log = os.path.join(path_WS, f'models/{Se_Ln["model alias"]}/code/snakemake/log/{Se_Ln["MdlN"]}.log')
            path_DAG = os.path.join(path_WS, f'models/{Se_Ln["model alias"]}/code/snakemake/DAG/DAG_{Se_Ln["MdlN"]}.png')
            print(f"\n{'-'*60}")
            print(f"-- {os.path.basename(path_Smk)}\n")

            try:
                if DAG:
                    sp.run(["snakemake", "--dag", "-s", path_Smk, "--cores", str(cores), '|', 'dot', '-Tpng', '-o', f'{path_DAG}'], shell=True, check=True)
                with open(path_log, 'w') as f:
                    sp.run(["snakemake", "-p", "-s", path_Smk, "--cores", str(cores)], check=True, stdout=f, stderr=f) # Run snakemake and write output to log file
                print(f"✅")
            except sp.CalledProcessError as e:
                print(f"❌: {e}")
            print(f"{'-'*60}")
    print(Sign)

def reset_Sim(MdlN: str):
    """Resets the simulation by deleting all files in the Sim folder and clearing the log.""" #666 can later be improved by deleting PoP files too. But that's not needed for now.
    d_paths = get_MdlN_paths(MdlN) # Get default directories
    path_MdlN = d_paths['path_MdlN']

    if os.path.exists(path_MdlN):
        try:
            sp.run(f'rmdir /S /Q "{path_MdlN}"', shell=True) # Delete the entire Sim folder
            DF = pd.read_csv(path_log) # Read the log file
            DF[ DF['MdlN']!=MdlN ].to_csv(path_log, index=False) # Remove the log entry for this model
            print(f"✅ - {path_MdlN} has been removed and log has been cleared.")
        except:
            print(f"❌ - failed to reset {path_MdlN}.")
    else:
        print(f"❌ - {path_MdlN} does not exist. No need to reset.")
        exit(1)
        
# Explore ---------------------------------------------------------------------------------
def Sim_Cfg(*l_MdlN, path_NP=r'C:\Program Files\Notepad++\notepad++.exe'):
    print(f"\n{'-'*100}\nOpening all configuration files for specified runs with the default program.\nIt's assumed that Notepad++ is installed in: {path_NP}.\nIf false, provide the correct path to Notepad++ (or another text editor) as the last argument to this function.\n")
    
    l_keys = ['path_Smk_S', 'path_BAT_S', 'path_INI_S', 'path_PRJ_S']
    l_paths = [get_MdlN_paths(MdlN) for MdlN in l_MdlN]
    l_files = [paths[k] for k in l_keys for paths in l_paths]
    sp.Popen([path_NP] + l_files)
    for f in l_files:
        print(f'\u2713 - {f}')
#---------------------------------------------------------------------------------
