## --- Imports ---
import WS_Mdl as WS
from WS_Mdl import Up_log
import os
from snakemake.io import temp
from datetime import datetime as dt
import pathlib

## --- Variables ---
MdlN = "NBr11"
Mdl = ''.join([i for i in MdlN if i.isalpha()])

path_Mdl = os.path.join(WS.path_WS, f'models/{Mdl}') 
workdir: path_Mdl
path_Smk = os.path.join(path_Mdl, 'code/snakemake')
path_temp = os.path.join(path_Smk, 'temp')
path_WS_lib = os.path.join(WS.path_WS, 'code/WS_Mdl/WS_Mdl.py')
path_Sim = os.path.join(path_Mdl, 'Sim')
path_MdlN = os.path.join(path_Sim, f'{MdlN}')
path_OBS, path_NAM = [os.path.join(path_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS', f'{MdlN}.NAM']]
path_HED, path_CBC = [os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]
path_LST_Sim = os.path.join(path_MdlN, 'MFSIM.NAM')

Log_init_done = f"{path_Smk}/temp/Log_init_done_{MdlN}"
log_SP_done = f"{path_Smk}/temp/log_SP_done_{MdlN}"

## --- Rules ---
rule all: # Final rule
    input:
        # path_LST_Sim,
        path_OBS,
        path_NAM,
        f"Sim/{MdlN}/RUN.BAT"

# -- PrP --
rule log_Init: # Sets status to running. Has to complete before anything else.
    input:
        path_WS_lib
    output:
        temp(Log_init_done)
    run:
        Up_log(MdlN, {  'End Status': 'Running',
                        'PrP start DT': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dir': path_Sim})
        pathlib.Path(output[0]).touch()

rule log_SP_dates: # Reads SP date range from INI file and writes the to log. 
    input:
        path_WS_lib,
        Log_init_done
    output:
        temp(log_SP_done)
    run:
        d_INI = WS.INI_to_d(WS.get_MdlN_paths(MdlN)['path_INI_S'])
        Up_log(MdlN, {'1st SP date': dt.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"), 'last SP date': dt.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        pathlib.Path(output[0]).touch()

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_SP_done,
        BAT = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.bat",
        INI = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.ini",
        PRJ = f"In/PRJ/{MdlN}.prj"
    output:
        BAT_RUN = f"Sim/{MdlN}/RUN.BAT"
    shell:
        "call {input.BAT}"
    ## Mdl_Prep Ins point to a lot of other files. Technically, all of them should be in the Ins. Practically, they don't. That is because Ins from previous runs aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

# -- PrSimP --
rule add_OBS:
    input:
        path_WS_lib,
        rules.Mdl_Prep.output.BAT_RUN
    output:
        path_OBS,
        path_NAM
    run:
        # WS.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")
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

# -- Sim ---
# rule Sim: # Runs the simulation via BAT file.
#     input:
#         path_OBS
#     output:
#         path_LST_Sim
#     run:
#         os.chdir(path_MdlN)
#         DT_Sim_Start = dt.now().strftime("%Y-%m-%d %H:%M:%S")
#         Up_log(MdlN, {'Sim start DT': DT_Sim_Start})
#         shell(f"{path_MdlN}/RUN.BAT")
#         Up_log(MdlN, {  'Sim end DT': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'Sim Duration': str(dt.now() - dt.strptime(DT_Sim_Start, "%Y-%m-%d %H:%M:%S")),
#                         'End Status': 'Completed'})

# --- Junkyard ---
# ## Can be replaced by starting PowerShell with this profile, like I've set-up Double Commander to do.
# rule activate_env:
#     shell:
#         activate WS

# rule make_DAG: # Creates a DAG of the workflow.
#     input:
#         Log_init_done
#     output:
#         f"{path_Smk}/DAG_{MdlN}.png"
#     run:
#         shell(f"snakemake --dag -p --dryrun -s {os.path.join(path_Smk, f'{MdlN}.smk')} --cores 1 | dot -Tpng -o {output}")