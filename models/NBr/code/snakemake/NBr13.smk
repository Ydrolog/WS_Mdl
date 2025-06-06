# --- Imports ---
# *** enable all #1 lines and disable/comment out all P_LST_Sim lines before re-runing ***

from WS_Mdl.utils import Up_log, Pa_WS, INI_to_d, get_elapsed_time_str, get_MdlN_paths
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM
import WS_Mdl.geo as G
from snakemake.io import temp
from datetime import datetime as DT
import pathlib
import os
import shutil as sh
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- Variables ---

## Options
MdlN        =   "NBr13"
MdlN_B_RIV  =   'NBr1'     # Baseline for RIV files
Mdl         =   U.get_Mdl(MdlN)
rules    =   "(L == 1)"

## Paths
Pa_Mdl          =   PJ(Pa_WS, f'models/{Mdl}') 
workdir:            Pa_Mdl
Pa_Smk          =   PJ(Pa_Mdl, 'code/snakemake')
Pa_temp         =   PJ(Pa_Smk, 'temp')
Pa_Sim          =   PJ(Pa_Mdl, 'Sim')
Pa_MdlN         =   PJ(Pa_Sim, f'{MdlN}')
Pa_BAT_RUN      =   PJ(Pa_MdlN, 'RUN.BAT')
Pa_OBS, Pa_NAM  =   [PJ(Pa_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS', f'{MdlN}.NAM']]
Pa_HED, Pa_CBC  =   [PJ(Pa_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]
Pa_LST_Sim      =   PJ(Pa_MdlN, 'mfsim.lst')

## Temp files (for completion validation)
log_Init_done       =   f"{Pa_Smk}/temp/Log_init_done_{MdlN}"
log_Sim_done        =   f"{Pa_Smk}/temp/Log_Sim_done_{MdlN}"
log_PRJ_to_TIF_done =   f"{Pa_Smk}/temp/Log_PRJ_to_TIF_done_{MdlN}"
log_GXG_done        =   f"{Pa_Smk}/temp/Log_GXG_done_{MdlN}"
log_Up_MM_done      =   f"{Pa_Smk}/temp/Log_Up_MM_done_{MdlN}"

## RIV Options
Pa_RIV        = PJ(Pa_Mdl, 'In/RIV')
Pa_RIV_MdlN   = PJ(Pa_RIV, f'{MdlN}')
l_In_PrP_RIV    = [str(file) for file in pathlib.Path(Pa_RIV).glob("*.idf") if "RIV_Stg" in file.name and MdlN_B_RIV in file.name]
l_Out_PrP_RIV   = [PJ(Pa_RIV_MdlN, PBN(i).replace(MdlN_B_RIV, MdlN)) for i in l_In_PrP_RIV]

# --- Rules ---

def fail(job, excecution):
    Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'End Status': 'Failed'})
onerror: fail


rule all: # Final rule
    input:
        Pa_LST_Sim,
        #1 log_Sim_done,  # This should have been enabled based on NBr16+ Smk methodology, but I don't want to re-run the Sim
        log_Up_MM_done
        
## -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init_done)
    run:
        import socket
        device = socket.gethostname()
        d_INI = INI_to_d(get_MdlN_paths(MdlN)['Pa_INI'])
        Up_log(MdlN, {  'End Status':       'Running',
                        'PrP start DT':     DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Sim device name":  device,
                        'Sim Dir':          Pa_Sim,
                        '1st SP date':      DT.strptime(d_INI['SDATE'], "%Y%m%d").strftime("%Y-%m-%d"),
                        'last SP date':     DT.strptime(d_INI['EDATE'], "%Y%m%d").strftime("%Y-%m-%d")})
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule PrP_RIV:
    input: # This input should never change.
        l_In_PrP_RIV
    output:
        l_Out_PrP_RIV
    script:
        PJ(Pa_WS, f'code/PrP/edit_RIV/edit_RIV_{MdlN}.py')

rule Mdl_Prep: # Prepares Sim Ins (from Ins) via BAT file.
    input:
        log_Init_done,
        l_Out_PrP_RIV,
        BAT = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.bat",
        INI = f"code/Mdl_Prep/Mdl_Prep_{MdlN}.ini",
        PRJ = f"In/PRJ/{MdlN}.prj"
    output:
        Pa_BAT_RUN
        # Pa_NAM
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
        #1 Up_log(MdlN, {  'Add OBS start DT': DT.now().strftime("%Y-%m-%d %H:%M:%S")})

## -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        Pa_OBS
    output:
        Pa_LST_Sim
        #1 log_Sim_done,  # This should have been enabled based on NBr16+ Smk methodology, but I don't want to re-run the Sim

    run:
        os.chdir(Pa_MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT': DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        shell(Pa_BAT_RUN)
        #1 pathlib.Path(output[0]).touch() 
        Up_log(MdlN, {  'Sim end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur':      get_elapsed_time_str(DT_Sim_Start),
                        'End Status':   'Completed'})

## -- PoP ---
rule PRJ_to_TIF:
    input:
        Pa_LST_Sim
        #1 log_Sim_done,  # This should have been enabled based on NBr16+ Smk methodology, but I don't want to re-run the Sim
    output:
        temp(log_PRJ_to_TIF_done)
    run:
        G.PRJ_to_TIF(MdlN) # First convert new Ins to TIF
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule GXG:
    input:
        Pa_LST_Sim
        #1 log_Sim_done,  # This should have been enabled based on NBr16+ Smk methodology, but I don't want to re-run the Sim
    output:
        temp(log_GXG_done)
    run:
        G.HD_IDF_GXG_to_TIF(MdlN)
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.

rule Up_MM:
    input:
        log_PRJ_to_TIF_done,
        log_GXG_done
    output:
        log_Up_MM_done
    run:
        G.Up_MM(MdlN)                   # Update MM 
        Up_log(MdlN, {  'PoP end DT':   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'End Status':   'PoPed'}) # Update log
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.


# --- Junkyard ---
# ## Can be replaced by starting PowerShell with this profile, like I've set-up Double Commander to do.
# rule activate_env:
#     shell:
#         activate WS