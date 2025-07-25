## --- Imports ---
import WS_Mdl as WS
from WS_Mdl import Up_log
from snakemake.io import temp
from datetime import datetime as DT
import pathlib
import os

## --- Variables ---
MdlN    =   "NBr11"
Mdl     =   ''.join([i for i in MdlN if i.isalpha()])

path_Mdl            =   os.path.join(WS.path_WS, f'models/{Mdl}') 
workdir:                path_Mdl
path_Smk            =   os.path.join(path_Mdl, 'code/snakemake')
path_temp           =   os.path.join(path_Smk, 'temp')
path_WS_lib         =   os.path.join(WS.path_WS, 'code/WS_Mdl/WS_Mdl.py')
path_Sim            =   os.path.join(path_Mdl, 'Sim')
path_MdlN           =   os.path.join(path_Sim, f'{MdlN}')
path_BAT_RUN        =   os.path.join(path_MdlN, 'RUN.BAT')
path_OBS, path_NAM  =   [os.path.join(path_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS', f'{MdlN}.NAM']]
path_HED, path_CBC  =   [os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]
path_LST_Sim        =   os.path.join(path_MdlN, 'mfsim.lst')

log_Init_done = f"{path_Smk}/temp/Log_init_done_{MdlN}"

## --- Rules ---
rule all: # Final rule
    input:
        path_LST_Sim
        
# -- PrP --
rule log_Init: # Sets status to running, and writes other info about therun. Has to complete before anything else.
    output:
        temp(log_Init_done)
    run:
        import socket
        device = socket.gethostname()
        d_INI = WS.INI_to_d(WS.get_MdlN_paths(MdlN)['path_INI_S'])
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
    ## Mdl_Prep Ins point to a lot of other files. Technically, all of them should be in the Ins. Practically, they don't. That is because Ins from previous runs aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

# -- PrSimP --
rule add_OBS:
    input:
        path_WS_lib,
        path_BAT_RUN
    output:
        path_OBS
    run:
        WS.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")
        # Up_log(MdlN, {  'Add OBS start DT': DT.now().strftime("%Y-%m-%d %H:%M:%S")})

# -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        path_OBS
    output:
        path_LST_Sim
    run:
        os.chdir(path_MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {'Sim start DT': DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        shell(path_BAT_RUN)
        Up_log(MdlN, {  'Sim end DT': DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur': WS.get_elapsed_time_str(DT_Sim_Start),
                        'End Status': 'Completed'})

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