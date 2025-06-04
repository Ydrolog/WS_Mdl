## --- Imports ---
from WS_Mdl.utils import Up_log, path_WS, INI_to_d, get_elapsed_time_str, get_MdlN_paths
import WS_Mdl.utils as U
import WS_Mdl.utils_imod as UIM
import WS_Mdl.geo as G
from snakemake.io import temp
from datetime import datetime as DT
import pathlib
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

## --- Variables ---
MdlN    =   "NBr15"
Mdl     =   ''.join([i for i in MdlN if i.isalpha()])

path_Mdl            =   os.path.join(path_WS, f'models/{Mdl}') 
workdir:                path_Mdl
path_Smk            =   os.path.join(path_Mdl, 'code/snakemake')
path_temp           =   os.path.join(path_Smk, 'temp')
path_Sim            =   os.path.join(path_Mdl, 'Sim')
path_MdlN           =   os.path.join(path_Sim, f'{MdlN}')
path_BAT_RUN        =   os.path.join(path_MdlN, 'RUN.BAT')
path_OBS, path_NAM  =   [os.path.join(path_MdlN, 'GWF_1', i) for i in [f'MODELINPUT/{MdlN}.OBS', f'{MdlN}.NAM']]
path_HED, path_CBC  =   [os.path.join(path_MdlN, 'GWF_1/MODELOUTPUT', i) for i in ['HEAD/HEAD.HED', 'BUDGET/BUDGET.CBC']]
path_LST_Sim        =   os.path.join(path_MdlN, 'mfsim.lst')

log_Init_done = f"{path_Smk}/temp/Log_init_done_{MdlN}"
log_Vis_Ins_done = f"{path_Smk}/temp/Log_Vis_Ind_done_{MdlN}"

## --- Rules ---
rule all: # Final rule
    input:
        path_LST_Sim,
        log_Vis_Ins_done
        
# -- PrP --
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
    ## Mdl_Prep Ins point to a lot of other files. Technically, all of them should be in the Ins. Practically, they don't. That is because Ins from previous runs aren't meant to be edited, as they're stamped with a MdlN. If one of the Ins that is new for this run is changed, then the script that edits that In Fi should be part of this snakemake file too.

# -- PrSimP --
rule add_OBS:
    input:
        path_BAT_RUN
    output:
        path_OBS
    run:
        UIM.add_OBS(MdlN, "BEGIN OPTIONS\n\tDIGITS 6\nEND OPTIONS")

# -- Sim ---
rule Sim: # Runs the simulation via BAT file.
    input:
        path_OBS
    output:
        path_LST_Sim
    run:
        os.chdir(path_MdlN) # Change directory to the model folder.
        DT_Sim_Start = DT.now()
        Up_log(MdlN, {  'Sim start DT'  :   DT_Sim_Start.strftime("%Y-%m-%d %H:%M:%S")})
        shell(path_BAT_RUN)
        Up_log(MdlN, {  'Sim end DT'    :   DT.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Sim Dur'       :   get_elapsed_time_str(DT_Sim_Start),
                        'End Status'    :   'Completed'})

rule PRJ_to_TIF:
    input:
        path_LST_Sim
    output:
        log_Vis_Ins_done
    run:
        G.PRJ_to_TIF(MdlN)
        pathlib.Path(output[0]).touch() # Create the file to mark the rule as done.



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
