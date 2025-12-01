# --- Imports ---
from WS_Mdl.utils import Up_log, Pa_WS, INI_to_d, get_elapsed_time_str
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM
import WS_Mdl.geo as G
from snakemake.io import temp
from datetime import datetime as DT
import pathlib
import subprocess as sp
import os
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import shutil as sh
import re
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONUNBUFFERED"] = "1"
from filelock import FileLock as FL

# --- Variables ---

## Options
MdlN        =   "NBr41"
#MdlN_SFR_OBS_Src =   'NBr25'
PP_rules       =   "(L == 1)"
#MdlN_B          =   U.get_MdlN_paths(MdlN)['MdlN_B']


## Paths
Mdl             =   U.get_Mdl(MdlN)
d_Pa            =   U.get_MdlN_Pa(MdlN)
Pa_Mdl          =   d_Pa['Pa_Mdl']
workdir:            Pa_Mdl
Pa_temp         =   d_Pa['Smk_temp']
Pa_Sim          =   d_Pa['Sim']
Pa_MdlN         =   PJ(Pa_Sim, f'{MdlN}')
#Pa_BAT_RUN      =   PJ(Pa_MdlN, 'RUN.BAT')
Pa_TOML         =   d_Pa['TOML']
Pa_OBS, Pa_NAM  =   [PJ(Pa_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS6', f'{MdlN}.NAM']]
#Pa_SFR_Src, Pa_SFR_Dst  =   PJ(Pa_Mdl, f"In/SFR/{MdlN}/{MdlN}.SFR6"), PJ(Pa_MdlN, f"GWF_1/MODELINPUT/{MdlN}.SFR6")
#Pa_SFR_OBS_Src, Pa_SFR_OBS_Dst  =   PJ(Pa_Mdl, f"In/OBS/SFR/{MdlN_SFR_OBS_Src}/{MdlN_SFR_OBS_Src}.SFR.OBS6"), PJ(Pa_MdlN, f"GWF_1/MODELINPUT/{MdlN}.SFR.OBS6")
Pa_HED, Pa_CBC  =   [PJ(Pa_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]
#Pa_MVR_Src, Pa_MVR_Dst  =   PJ(Pa_Mdl, f"In/MVR/{MdlN}/{MdlN}.MVR"), PJ(Pa_MdlN, f"GWF_1/MODELINPUT/{MdlN}.MVR")
l_Fi_to_git = [PJ(Pa_WS, i) for i in ['code/pixi.toml', 'code/pixi.lock', 'code/WS_Mdl']] # If any of these code files changes, the env needs to be frozen.

## Temp files (for completion validation)
log_Init           =   f"{Pa_temp}/Log_init_{MdlN}"
log_Sim            =   f"{Pa_temp}/Log_Sim_{MdlN}"
log_PRJ_to_TIF     =   f"{Pa_temp}/Log_PRJ_to_TIF_{MdlN}"
log_GXG            =   f"{Pa_temp}/Log_GXG_{MdlN}"
log_Up_MM          =   f"{Pa_temp}/Log_Up_MM_{MdlN}"
log_freeze_env     =   f"{Pa_temp}/Log_freeze_env_{MdlN}"
log_MVR_OPTIONS    =   f"{Pa_temp}/Log_MVR_OPTIONS{MdlN}"


# --- Rules ---

def fail(job, excecution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        # log_Sim,
        # log_Up_MM,
        Pa_TOML,
        log_freeze_env

rule debug_env:
    output:
        temp(f"{Pa_temp}/Log_debug_env_{MdlN}")
    run:
        import sys, subprocess
        print("-/\/\/\/\= Python executable:", sys.executable)
        subprocess.run(["which", "python"])
        subprocess.run(["which", "snakemake"])


## -- PrP --
rule log_Init: # Sets status to running, and writes other info about the Sim. Has to complete before anything else.
    input:
        f"{Pa_temp}/Log_debug_env_{MdlN}"
    output:
        temp(log_Init)
    run:
        import socket
        device = socket.gethostname()
        d_INI = INI_to_d(d_Pa['INI'])
        Up_log(MdlN, {  'End Status':       'Running',
                        'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim device name':  device,
                        'Sim Dir':          Pa_Sim,
                        '1st SP date':      DT.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date':     DT.strptime(d_INI['EDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule freeze_pixi_env:
    input:
        l_Fi_to_git
    output:
        temp(log_freeze_env)
    run:
        git_hash, git_tag = U.freeze_pixi_env(MdlN)
        Up_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule iMPy_Mdl_Prep: # Prepares Sim Ins (from Ins) via iMOD python. iMOD python still uses an INI and a PRJ file.
    input:
        log_Init,
        BAT = d_Pa['BAT'],
        INI = d_Pa['INI'],
        PRJ = d_Pa['PRJ']
    output:
        Pa_TOML
    shell:
        "pixi run UIM.Mdl_Prep(MdlN, verbose=True)"
    ## Mdl_Prep Ins (mainly the PRJ) point to a lot of other files. Technically, all of them should be in the Ins of this rule. Practically, they don't need to be. That is because Ins from previous Sims aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

# ## -- PrSimP --
# rule add_OBS:
#     input:
#         Pa_BAT_RUN
#     output:
#         Pa_OBS
#     run:
#         UIM.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")

# ## -- Sim ---
# rule Sim: # Runs the simulation via BAT file.
#     input:
#         Pa_OBS
#     output:
#         temp(log_Sim)
#     run:
#         os.chdir(Pa_MdlN) # Change directory to the model folder.
#         DT_Sim_Start = DT.now()
#         Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
#         shell(Pa_BAT_RUN)
#         pathlib.Path(output[0]).touch() 
#         Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
#                         'End Status'    :   'Completed'})

# ## -- PoP ---
# rule PRJ_to_TIF:
#     input:
#         log_Sim
#     output:
#         temp(log_PRJ_to_TIF)
#     run:
#         G.PRJ_to_TIF(MdlN)
#         pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

# rule GXG:
#     input:
#         log_Sim
#     output:
#         temp(log_GXG)
#     run:
#         G.HD_IDF_GXG_to_TIF(MdlN, rules=PP_rules)
#         pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

# rule Up_MM:
#     input:
#         log_PRJ_to_TIF,
#         log_GXG
#     output:
#         log_Up_MM
#     run:
#         G.Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)     # Update MM 
#         Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'End Status':   'PoPed'}) # Update log
#         pathlib.Path(output[0]).touch()     # Create the file to mark the rule as done.