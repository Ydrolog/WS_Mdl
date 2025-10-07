## --- Imports ---

from WS_Mdl.utils import Up_log, Pa_WS, INI_to_d, get_elapsed_time_str, get_MdlN_paths
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
MdlN        =   "NBr34"
MdlN_MM_B   =   'NBr17'
Mdl         =   ''.join([i for i in MdlN if i.isalpha()])
rule_       =   "(L == 1)"

## Paths
Pa_Mdl          =   PJ(Pa_WS, f'models/{Mdl}') 
workdir:            Pa_Mdl
Pa_Smk          =   PJ(Pa_Mdl, 'code/snakemake')
Pa_temp         =   PJ(Pa_Smk, 'temp')
Pa_Sim          =   PJ(Pa_Mdl, 'Sim')
Pa_MdlN         =   PJ(Pa_Sim, f'{MdlN}')
Pa_BAT_RUN      =   PJ(Pa_MdlN, 'RUN.BAT')
Pa_OBS, Pa_NAM  =   [PJ(Pa_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS6', f'{MdlN}.NAM']]
Pa_HED, Pa_CBC  =   [PJ(Pa_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]

git_hash = shell("git rev-parse HEAD", read=True).strip()
git_tag  = shell("git describe --tags --always", read=True, allow_error=True).strip() or "no_tag"

## Temp files (for completion validation)
log_Init           =   f"{Pa_Smk}/temp/Log_init_{MdlN}"
log_Sim            =   f"{Pa_Smk}/temp/Log_Sim_{MdlN}"
log_PRJ_to_TIF     =   f"{Pa_Smk}/temp/Log_PRJ_to_TIF_{MdlN}"
log_GXG            =   f"{Pa_Smk}/temp/Log_GXG_{MdlN}"
log_Up_MM          =   f"{Pa_Smk}/temp/Log_Up_MM_{MdlN}"
log_freeze_env          =   f"{Pa_Smk}/temp/Log_freeze_env_{MdlN}"

# --- Rules ---

def fail(job, excecution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        log_Sim,
        log_Up_MM,
        log_freeze_env
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init)
    run:
        import socket
        device = socket.gethostname()
        d_INI = INI_to_d(get_MdlN_paths(MdlN)['INI'])

        Up_log(MdlN, {  'End Status':       'Running',
                        'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Sim device name":  device,
                        'Sim Dir':          Pa_Sim,
                        '1st SP date':      DT.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date':     DT.strptime(d_INI['EDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.
        # Move this to a new rule later, when you've made it safe for multiple rules to write to the same file. Low priority...
        Up_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info

rule freeze_env_to_yml:
    output: 
        log_freeze_env
    run:
        try:
            sp.run(["python", PJ(Pa_WS,"code/Env/freeze_env_WS.py"), "--MdlN", MdlN], check=True)
            pathlib.Path(output[0]).touch()
        except Exception:
            print(f'Error will be written to {output[0]}')
            open(output[0], "w").close()

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init,
        BAT = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.bat",
        INI = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.ini",
        PRJ = f"In/PRJ/{MdlN}.prj"
    output:
        Pa_BAT_RUN
    shell:
        "call {input.BAT}"
    ## Mdl_Prep Ins (mainly the PRJ) point to a lot of other files. Technically, all of them should be in the Ins of this rule. Practically, they don't need to be. That is because Ins from previous Sims aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

## -- PrSimP --
rule add_OBS:
    input:
        Pa_BAT_RUN
    output:
        Pa_OBS
    run:
        UIM.add_OBS(MdlN, iMOD5=True)

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        Pa_OBS
    output:
        temp(log_Sim)
    run:
        os.chdir(Pa_MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        shell(Pa_BAT_RUN)
        pathlib.Path(output[0]).touch() 
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                        'End Status'    :   'Completed'})

## -- PoP ---
rule PRJ_to_TIF:
    input:
        log_Sim
    output:
        temp(log_PRJ_to_TIF)
    run:
        G.PRJ_to_TIF(MdlN)
        Up_log(MdlN, {  'PRJ_to_TIF':   1})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        log_Sim
    output:
        temp(log_GXG)
    run:
        G.HD_IDF_GXG_to_TIF(MdlN, rules=rules)
        Up_log(MdlN, {  'GXG':   rule_})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF,
        log_GXG
    output:
        log_Up_MM
    run:
        G.Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)     # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed',
                        'Up_MM'     :   1}) # Update log
        pathlib.Path(output[0]).touch()     # Create the file to mark the rule as done.