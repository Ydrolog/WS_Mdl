# --- Imports ---

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
MdlN            =   "NBr28"
MdlN_SFR_OBS_Src=   "NBr25"
MdlN_MM_B       =   'NBr18'
Mdl             =   ''.join([i for i in MdlN if i.isalpha()])
rule_           =   "(L == 1)"
MdlN_B          =   U.get_MdlN_paths(MdlN)['MdlN_B']

## Paths
Pa_Mdl                  =   PJ(Pa_WS, f'models/{Mdl}') 
workdir:                    Pa_Mdl
Pa_Smk                  =   PJ(Pa_Mdl, 'code/snakemake')
Pa_temp                 =   PJ(Pa_Smk, 'temp')
Pa_Sim                  =   PJ(Pa_Mdl, 'Sim')
Pa_MdlN                 =   PJ(Pa_Sim, f'{MdlN}')
Pa_BAT_RUN              =   PJ(Pa_MdlN, 'RUN.BAT')
Pa_OBS, Pa_NAM          =   [PJ(Pa_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS6', f'{MdlN}.NAM']]
Pa_SFR_Src, Pa_SFR_Dst  =   PJ(Pa_Mdl, f"In/SFR/{MdlN_B}/{MdlN_B}.SFR6"), PJ(Pa_MdlN, f"GWF_1/MODELINPUT/{MdlN}.SFR6")
Pa_SFR_OBS_Src, Pa_SFR_OBS_Dst  =   PJ(Pa_Mdl, f"In/OBS/SFR/{MdlN_B}/{MdlN_B}.SFR.OBS6"), PJ(Pa_MdlN, f"GWF_1/MODELINPUT/{MdlN}.SFR.OBS6")
Pa_HED, Pa_CBC          =   [PJ(Pa_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]

git_hash = shell("git rev-parse HEAD", read=True).strip()
git_tag  = shell("git describe --tags --exact-match", read=True, allow_error=True).strip() or "no_tag"

## Temp files (for completion validation)
log_Init_done           =   f"{Pa_Smk}/temp/Log_init_done_{MdlN}"
log_Sim_done            =   f"{Pa_Smk}/temp/Log_Sim_done_{MdlN}"
log_PRJ_to_TIF_done     =   f"{Pa_Smk}/temp/Log_PRJ_to_TIF_done_{MdlN}"
log_GXG_done            =   f"{Pa_Smk}/temp/Log_GXG_done_{MdlN}"
log_Up_MM_done          =   f"{Pa_Smk}/temp/Log_Up_MM_done_{MdlN}"
log_freeze_env_done     =   f"{Pa_Smk}/temp/Log_freeze_env_done_{MdlN}"


# --- Rules ---

def fail(job, excecution): # Gets activated if any rule fails.
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        log_Sim_done,
        log_Up_MM_done,
        log_freeze_env_done
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init_done)
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

        # Move this to a new ule later, when you've made it safe for multiple rules to write tothe same file. Low priority...
        Up_log(MdlN, {  'Git hash': git_hash,
                        'Git tag': git_tag}) # Log git info

rule freeze_env_to_yml:
    output: log_freeze_env_done
    run:
        sp.run(
            ["python", PJ(Pa_WS,"code/Env/freeze_env_WS.py"), "--MdlN", MdlN],
            check=True
        )
        pathlib.Path(output[0]).touch()

        
rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init_done,
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
        UIM.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")

rule add_SFR_copy:
    input:
        Pa_BAT_RUN,
        Pa_SFR_Src
    output:
        Pa_SFR_Dst
    run:
        sh.copy2(Pa_SFR_Src, Pa_SFR_Dst)
        lock = FL(Pa_NAM + '.lock')  # Create a file lock to prevent concurrent writes
        with lock, open(Pa_NAM, 'r+') as f:
            l_lines     = f.readlines()
            l_lines[-1] = f" SFR6 ./GWF_1/MODELINPUT/{MdlN}.SFR6 SFR6\n"
            f.seek(0); f.truncate()
            for i in l_lines:
                f.write(i)
            f.write('END PACKAGES')

rule add_SFR_OBS_copy: # By copying file.
    input:
        Pa_BAT_RUN,
        Pa_SFR_OBS_Src
    output:
        Pa_SFR_OBS_Dst
    run:
        sh.copy2(Pa_SFR_OBS_Src, Pa_SFR_OBS_Dst)

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        Pa_OBS,
        Pa_SFR_Dst,
        Pa_SFR_OBS_Dst
    output:
        temp(log_Sim_done)
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
        G.HD_IDF_GXG_to_TIF(MdlN, rules=rule_)
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF_done,
        log_GXG_done
    output:
        log_Up_MM_done
    run:
        G.Up_MM(MdlN, MdlN_MM_B=MdlN_MM_B)     # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed'}) # Update log
        pathlib.Path(output[0]).touch()     # Create the file to mark the rule as done.