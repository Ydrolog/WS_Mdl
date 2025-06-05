# --- Imports ---
from WS_Mdl.utils import Up_log, path_WS, INI_to_d, get_elapsed_time_str, get_MdlN_paths
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM
import WS_Mdl.geo as G
from snakemake.io import temp
from datetime import datetime as DT
import pathlib
import os
import shutil as sh
import re
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONUNBUFFERED"] = "1"

# --- Variables ---

## Options
MdlN        =   "NBr16"
MdlN_B      =   'Nbr15'
MdlN_MM_B   =   'NBr12'
Mdl         =   ''.join([i for i in MdlN if i.isalpha()])
DF_rules    =   "(L == 1)"

## Paths
path_Mdl                        =   os.path.join(path_WS, f'models/{Mdl}') 
workdir:                            path_Mdl
path_Smk                        =   os.path.join(path_Mdl, 'code/snakemake')
path_temp                       =   os.path.join(path_Smk, 'temp')
path_Sim                        =   os.path.join(path_Mdl, 'Sim')
path_MdlN                       =   os.path.join(path_Sim, f'{MdlN}')
path_BAT_RUN                    =   os.path.join(path_MdlN, 'RUN.BAT')
path_OBS, path_NAM, path_SFR    =   [os.path.join(path_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS6', f'{MdlN}.NAM', f'MODELINPUT/{MdlN}.SFR6']]
path_SFR_B                      =   path_SFR.replace(MdlN, MdlN_B)
path_HED, path_CBC              =   [os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]

## Temp files (for completion validation)
log_Init_done       =   f"{path_Smk}/temp/Log_init_done_{MdlN}"
log_Sim_done        =   f"{path_Smk}/temp/Log_Sim_done_{MdlN}"
log_PRJ_to_TIF_done =   f"{path_Smk}/temp/Log_PRJ_to_TIF_done_{MdlN}"
log_GXG_done        =   f"{path_Smk}/temp/Log_GXG_done_{MdlN}"
log_Up_MM_done      =   f"{path_Smk}/temp/Log_Up_MM_done_{MdlN}"


# --- Rules ---
rule all: # Final rule
    input:
        log_Sim_done,
        log_Up_MM_done
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init_done)
    run:
        import socket
        device = socket.gethostname()
        d_INI = INI_to_d(get_MdlN_paths(MdlN)['path_INI'])
        Up_log(MdlN, {  'End Status':       'Running',
                        'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Sim device name":  device,
                        'Sim Dir':          path_Sim,
                        '1st SP date':      DT.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date':     DT.strptime(d_INI['EDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init_done,
        BAT = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.bat",
        INI = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.ini",
        PRJ = f"In/PRJ/{MdlN}.prj"
    output:
        path_BAT_RUN
        # path_NAM
    shell:
        "call {input.BAT}"
    ## Mdl_Prep Ins (mainly the PRJ) point to a lot of other files. Technically, all of them should be in the Ins of this rule. Practically, they don't need to be. That is because Ins from previous Sims aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

## -- PrSimP --
rule add_OBS:
    input:
        path_BAT_RUN
    output:
        path_OBS
    run:
        UIM.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")

rule add_SFR:
    input:
        path_BAT_RUN
    output:
        path_SFR
    run:
        sh.copy2(path_SFR_B, path_SFR)
        with open(path_NAM, 'r') as f1:
            l_lines = f1.readlines()
        with open(path_NAM, 'w') as f2:
            for i in l_lines:
                f2.write( re.sub(MdlN_B, MdlN, i, flags=re.IGNORECASE))

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        path_OBS,
        path_SFR
    output:
        temp(log_Sim_done)
    run:
        os.chdir(path_MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        shell(path_BAT_RUN)
        pathlib.Path(output[0]).touch() 
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                        'End Status'    :   'Completed'})

## -- PoP ---
rule PRJ_to_TIF:
    input:
        log_Sim_done
    output:
        temp(log_PRJ_to_TIF_done)
    run:
        G.PRJ_to_TIF(MdlN)
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        log_Sim_done
    output:
        temp(log_GXG_done)
    run:
        G.HD_IDF_GXG_to_TIF(MdlN, DF_rules=DF_rules)
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF_done,
        log_GXG_done
    output:
        log_Up_MM_done
    run:
        G.Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)     # Update MM 
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        pathlib.Path(output[0]).touch()     # Create the file to mark the rule as done.


# rule fail: # Runs only if the Sim has failed, to update the log.
#     input:
#         path_LST_Sim
#     output:
#         temp(path_fail) # need to add this to ruleall
#     run:
#         Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'End Status': 'Failed'})
#         raise Exception("Simulation failed.")


# --- Junkyard ---
# ## Can be replaced by starting PowerShell with this profile, like I've set-up Double Commander to do.
# rule activate_env:
#     shell:
#         activate WS