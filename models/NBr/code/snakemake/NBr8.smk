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
BAT_RUN = os.path.join(path_MdlN, 'RUN.BAT')
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
        BAT_RUN
        
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
        BAT_RUN,
        path_NAM
    shell:
        "call {input.BAT}"
    ## Mdl_Prep Ins point to a lot of other files. Technically, all of them should be in the Ins. Practically, they don't. That is because Ins from previous runs aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

# -- PrSimP --
rule add_OBS:
    input:
        path_WS_lib,
        BAT_RUN
    output:
        path_OBS
    run:
        WS.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")

ruleorder: Mdl_Prep > add_OBS

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