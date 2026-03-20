import os
import shutil as sh
import stat
import subprocess as sp
import sys
from datetime import datetime as DT
from multiprocessing import Pool, cpu_count

import pandas as pd
from colored import attr, fg
from send2trash import send2trash
from WS_Mdl.core.log import DF_match_MdlN, r_RunLog, to_Se
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.path import MdlN_PaView, Pa_log_Cfg, Pa_log_Out, Pa_WS
from WS_Mdl.core.style import Sep, bold, dim, sprint, style_reset, warn


def S_from_B(MdlN: str, iMOD5=False):
    """Copies files that contain Sim options from the B Sim, renames them for the S Sim, and opens them in the default file editor. Assumes default WS_Mdl folder structure (as described in READ_ME.MD)."""

    sprint(Sep)
    M = Mdl_N(MdlN)
    MdlN_B = to_Se(MdlN)['B MdlN']
    M_B = Mdl_N(MdlN_B)

    # Keep explicit iMOD5 override behavior while defaulting to model-native paths.
    Pa = M.Pa if iMOD5 == (M.V == 'imod5') else MdlN_PaView(MdlN, iMOD5=iMOD5)
    Pa_B = M_B.Pa if iMOD5 == (M_B.V == 'imod5') else MdlN_PaView(MdlN_B, iMOD5=iMOD5)

    Pa_INI_B, Pa_INI = Pa_B.INI, Pa.INI
    Pa_BAT_B, Pa_BAT = Pa_B.BAT, Pa.BAT
    Pa_Smk_B, Pa_Smk = Pa_B.Smk, Pa.Smk
    Pa_PRJ_B, Pa_PRJ = Pa_B.PRJ, Pa.PRJ

    # Copy .INI, .bat, .prj and make default (those apply to every Sim) modifications
    for Pa_B, Pa_S in zip([Pa_Smk_B, Pa_BAT_B, Pa_INI_B], [Pa_Smk, Pa_BAT, Pa_INI]):
        try:
            if not Pa_S.exists():  # Replace the MdlN of with the new one, so that we don't have to do it manually.
                sh.copy2(Pa_B, Pa_S)
                with open(Pa_S, 'r') as f1:
                    contents = f1.read()
                with open(Pa_S, 'w') as f2:
                    f2.write(contents.replace(MdlN_B, MdlN))
                if Pa_B.suffix.lower() != '.bat':
                    os.startfile(
                        Pa_S
                    )  # Then we'll open it to make any other changes we want to make. Except if it's the BAT file
                sprint(f'🟢 - {Pa_S.name:20} created successfully! {dim}(copy of {Pa_B}){style_reset}')
            else:
                sprint(
                    f'🟡 - {Pa_S.name:20} already exists. If you want it to be replaced, you have to delete it manually before running this command.'
                )
        except Exception as e:
            sprint(f'🔴 - Error copying {Pa_B} to {Pa_S}: {e}')

    try:
        if not Pa_PRJ.exists():  # For the PRJ file, there is no default text replacement to be performed.
            sh.copy2(Pa_PRJ_B, Pa_PRJ)
            os.startfile(Pa_PRJ)  # Then we'll open it to make any other changes we want to make.
            sprint(f'🟢 - {Pa_PRJ.name:20} created successfully! {dim}(copy of {Pa_PRJ_B}){style_reset}')
        else:
            sprint(
                f'🟡 - {Pa_PRJ.name:20} already exists. If you want it to be replaced, you have to delete it manually before running this command.'
            )
    except Exception as e:
        sprint(f'🔴 - Error copying {Pa_PRJ_B} to {Pa_PRJ}: {e}')

    sprint(Sep)


def S_from_B_undo(MdlN: str):
    """Will undo S_from_B by deletting S files"""
    sprint(Sep)
    M = Mdl_N(MdlN)
    Pa = M.Pa
    Pa_Smk, Pa_BAT, Pa_INI, Pa_PRJ = Pa.Smk, Pa.BAT, Pa.INI, Pa.PRJ

    sprint(
        f'Are you sure you want to delete the Cfg files (.smk, .ini, .bat, .prj) for {MdlN}? (y/n):',
        style=warn,
    )
    confirm = input().strip().lower()
    if confirm == 'y':
        print()
        for Pa_S in [Pa_Smk, Pa_BAT, Pa_INI, Pa_PRJ]:
            Pa_S.unlink()  # Delete the S files
            sprint(f'🟢 - {Pa_S.name:20} deleted successfully!')

    sprint(Sep)


def RunSim(args):
    """Helper function that runs a single model's snakemake workflow."""
    _, Se_Ln, cores_per_Sim, generate_dag, no_temp = args
    M = Mdl_N(Se_Ln['MdlN'])
    Pa_Smk_log = M.Pa.Mdl / f'code/snakemake/log/{Se_Ln["MdlN"]}_{DT.now().strftime("%Y%m%d_%H%M%S")}.log'
    sprint(f'{fg("cyan")}{M.Pa.Smk.name}{attr("reset")}\n')

    try:
        if generate_dag:  # DAG parameter passed from RunMng
            cmd = (
                f'pixi run --manifest-path "{M.Pa.pixi}" snakemake --directory "{M.Pa.Mdl}" --dag -s "{M.Pa.Smk}" --cores {cores_per_Sim} '
                f'| pixi run --manifest-path "{M.Pa.pixi}" dot -Tpng -o "{M.Pa.Smk_DAG}"'
            )
            sp.run(cmd, shell=True, check=True)

        with open(Pa_Smk_log, 'w', encoding='utf-8-sig') as f:
            cmd = [
                'pixi',
                'run',
                '--manifest-path',
                M.Pa.pixi,
                'snakemake',
                '--directory',
                M.Pa.Mdl,
                '-p',
                '-s',
                M.Pa.Smk,
                '--cores',
                str(cores_per_Sim),
            ]

            if no_temp:
                cmd.append('--notemp')
            sp.run(cmd, shell=False, check=True, stdout=f, stderr=f)
        return (Se_Ln['MdlN'], True)
    except sp.CalledProcessError as e:
        return (Se_Ln['MdlN'], False, str(e))


def RunMng(cores=None, DAG: bool = True, Cct_Sims=None, no_temp: bool = True):
    """
    Read the RunLog, and for each queued model, run the corresponding Snakemake file.

    Parameters:
        cores: Number of cores to allocate to each Snakemake process
        DAG: Whether to generate a DAG visualization
        Cct_Sims: Number of models to run simultaneously (defaults to number of available cores)
    """

    os.chdir(Pa_WS)

    if cores is None:
        cores = max(
            cpu_count() - 2, 1
        )  # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    sprint(
        f'{Sep}RunMng initiated on {fg("cyan")}{str(DT.now()).split(".")[0]}{attr("reset")}. All Sims that are queued in the RunLog will be executed.\n'
    )

    sprint('Reading RunLog ...', end='')
    DF = pd.read_csv(Pa_log_Cfg)
    DF_q = DF.loc[
        ((DF['Start Status'] == 'Queued') & ((DF['End Status'].isna()) | (DF['End Status'] == 'Failed')))
    ]  # _q for queued. Only Run Queued runs that aren't running or have finished.
    sprint(' completed!\n')

    if not Cct_Sims:
        N_Sims = len(DF_q)
        Cct_Sims = max(
            min(N_Sims, cores), 1
        )  # Number of Sims to run simultaneously, limited by number of queued runs and available cores

    cores_per_Sim = cores // Cct_Sims  # Number of cores per Sim

    sprint(
        f'Found {fg("cyan")}{len(DF_q)} queued Sim(s){attr("reset")} in the RunLog. Will run {fg("cyan")}{Cct_Sims} Sim(s) simultaneously{attr("reset")}, using {bold}{cores_per_Sim} cores per Sim{style_reset}.\n'
    )

    if DF_q.empty:
        sprint('\n🟡🟡🟡 - No queued runs found in the RunLog.')
    else:
        # Prepare arguments for multiprocessing
        args = [(i, row, cores_per_Sim, DAG, no_temp) for i, row in DF_q.iterrows()]

        # Run models in parallel
        with Pool(processes=Cct_Sims) as pool:
            results = pool.map(RunSim, args)

        # Print results
        for result in results:
            if len(result) == 2:
                model_id, success = result
                if success:
                    sprint(f'🟢🟢 Model {model_id} completed successfully')
                else:
                    sprint(f'🔴🔴 Model {model_id} failed')
            else:
                model_id, success, error = result
                sprint(f'🔴🔴 Model {model_id} failed: {error}')

    sprint(Sep)


def reset_Sim(MdlN: str, ask_permission: bool = True, Pa_log_Out=Pa_log_Out, permanent_delete: bool = False):
    """
    Resets the simulation (like if it never happened, but with the files to recreate it still there.) by:
        1. Moving all files in the MldN folder (in the Sim folder) to recycling bin (or permanently deleting if permanent_delete=True).
        2. Clearing log.csv.
        3. Moving Smk temp files for MdlN to recycling bin (or permanently deleting if permanent_delete=True).
        4. Moving PoP folder for MdlN to recycling bin (or permanently deleting if permanent_delete=True).

    Parameters
    ----------
    MdlN : str
        Model name identifier
    ask_permission : bool, default=True
        Whether to ask for user confirmation before proceeding
    Pa_log_Out : str
        Path to the log CSV file
    permanent_delete : bool, default=False
        If True, files are permanently deleted. If False, files are moved to recycling bin.
    """

    sprint(Sep)
    if ask_permission:
        action = 'permanently delete' if permanent_delete else 'recycle'
        permission = (
            input(
                f'{warn}This will {action} the corresponding Sim/{MdlN} & PoP/Out/{MdlN} folders, and change the status of the corresponding line of log.csv to "removed_Out". Are you sure you want to proceed? (y/n):\n{style_reset}'
            )
            .strip()
            .lower()
        )
    else:
        permission = 'y'

    if permission == 'y':
        M = Mdl_N(MdlN)
        Pa_MdlN = M.Pa.MdlN
        DF = pd.read_csv(Pa_log_Out)  # Read the log file
        Pa_Smk_temp = M.Pa.Smk_temp
        l_temp = [p for p in Pa_Smk_temp.iterdir() if MdlN.lower() in p.name.lower()]

        if (
            Pa_MdlN.exists() or (MdlN.lower() in DF['MdlN'].str.lower().values) or l_temp or M.Pa.PoP_Out_MdlN.exists()
        ):  # Check if the Sim folder exists or if the MdlN is in the log file
            i = 0

            try:  # --- Remove Sim folder ---
                if not Pa_MdlN.exists():
                    raise FileNotFoundError(f'{Pa_MdlN} does not exist.')
                if permanent_delete:
                    sp.run(f'rmdir /S /Q "{Pa_MdlN}"', shell=True)  # Permanently delete the entire Sim folder
                    sprint('🟢 - Sim folder permanently deleted successfully.')
                else:
                    send2trash(Pa_MdlN)  # Move the entire Sim folder to recycling bin
                    sprint('🟢 - Sim folder moved to recycling bin successfully.')
                i += 1
            except Exception as e:
                action = 'permanently delete' if permanent_delete else 'move to recycling bin'
                sprint(f'🔴 - failed to {action} Sim folder: {e}')

            try:  # --- Remove log.csv entry ---
                DF[DF['MdlN'].str.lower() != MdlN.lower()].to_csv(
                    Pa_log_Out, index=False
                )  # Remove the log entry for this model
                sprint('🟢 - log.csv file updated successfully.')
                i += 1
            except Exception as e:
                sprint(f'🔴 - failed to update log.csv file: {e}')

            try:  # --- Remove temp Smk files ---
                if l_temp:
                    for Pa_temp in l_temp:
                        if permanent_delete:
                            Pa_temp.unlink()  # Permanently delete temp files
                        else:
                            send2trash(Pa_temp)  # Move temp files to recycling bin
                    action = 'permanently deleted' if permanent_delete else 'moved to recycling bin'
                    sprint(f'🟢 - Smk temp files {action} successfully.')
                    i += 1
                else:
                    sprint('🟡 - No Smk temp files found to delete.')
            except Exception as e:
                action = 'permanently delete' if permanent_delete else 'move to recycling bin'
                sprint(f'🔴 - failed to {action} Smk temp files: {e}')

            try:  # --- Remove PoP folder ---
                if not M.Pa.PoP_Out_MdlN.exists():
                    raise FileNotFoundError(f'{M.Pa.PoP_Out_MdlN} does not exist.')
                if permanent_delete:
                    sp.run(f'rmdir /S /Q "{M.Pa.PoP_Out_MdlN}"', shell=True)  # Permanently delete the PoP folder
                    sprint('🟢 - PoP Out folder permanently deleted successfully.')
                else:
                    send2trash(M.Pa.PoP_Out_MdlN)  # Move the entire PoP folder to recycling bin
                    sprint('🟢 - PoP Out folder moved to recycling bin successfully.')
                i += 1
            except Exception as e:
                action = 'permanently delete' if permanent_delete else 'move to recycling bin'
                sprint(f'🔴 - failed to {action} PoP Out folder: {e}')

            if i == 4:
                action = 'permanently deleted' if permanent_delete else 'moved to recycling bin'
                sprint(f'\n🟢🟢🟢 - ALL files were successfully {action}.')
            else:
                sprint(f'🟡🟡🟡 - {i}/4 sub-processes finished successfully.')
        else:
            sprint(
                '🔴🔴🔴 - Items do not exist (Sim folder, log entry, Smk log files, PoP Out folder). No need to reset.'
            )
    else:
        sprint('🔴🔴🔴 - Reset cancelled by user (you).')
    sprint(Sep)


def remove_Sim_Out(
    MdlN: str, Del_all: bool = False, ask_permission: bool = True, Pa_log=Pa_log_Out, permanent_delete: bool = False
):
    """
    Removes Sim Out, but not the PoP. Specifically:
        1. Moves all files in the MldN folder (inside the Sim folder) to recycling bin (or permanently deletes if permanent_delete=True).
        2. Changes log.csv status to "removed_Out".

    Parameters
    ----------
    MdlN : str
        Model name identifier
    ask_permission : bool, default=True
        Whether to ask for user confirmation before proceeding
    Pa_log : str
        Path to the log CSV file
    permanent_delete : bool, default=False
        If True, files are permanently deleted. If False, files are moved to recycling bin.
    """

    def _on_rm_error(func, path, exc_info):
        """
        Error handler for ``shutil.rmtree``.

        If the error is due to an access error (read only file)
        it attempts to add write permission and then retries.

        If the error is for another reason it re-raises the error.

        Usage : ``shutil.rmtree(path, onerror=_on_rm_error)``
        """
        # Is the error an access error?
        if issubclass(exc_info[0], PermissionError):
            path.chmod(stat.S_IWRITE)
            try:
                func(path)
            except Exception:
                try:
                    import time

                    time.sleep(0.1)
                    func(path)
                except Exception:
                    raise
        else:
            raise

    sprint(Sep)

    if Del_all:
        Del_text = 'All files will be removed (Del_all=True).'
    else:
        Del_text = 'Only large output files (.hds, .cbc, .grb in the MF folder; and MSW out folders) will be removed (Del_all=False).'

    # ---------- Permission ----------
    action = 'permanently delete' if permanent_delete else 'recycle'
    M = Mdl_N(MdlN)
    if ask_permission:
        permission = (
            input(
                f'{warn}This will {action} files in {Pa_WS}/models/{M.alias}/Sim/{MdlN} folder, and change the status of the corresponding line of log.csv.\n{Del_text}\nAre you sure you want to proceed? (y/n):\n{style_reset}'
            )
            .strip()
            .lower()
        )
    else:
        permission = 'y'

    # ---------- Remove + Update log ----------
    if permission == 'y':
        Pa = M.Pa
        Pa_MdlN = Pa.Pa_MdlN
        DF = pd.read_csv(Pa_log)  # Read the log file

        if Pa_MdlN.exists():
            i = 0
            if Del_all:
                try:  # --- Remove whole Sim folder ---
                    if not Pa_MdlN.exists():
                        raise FileNotFoundError(f'{Pa_MdlN} does not exist.')
                    if permanent_delete:
                        sp.run(f'rmdir /S /Q "{Pa_MdlN}"', shell=True)  # Permanently delete the entire Sim folder
                    else:
                        send2trash(Pa_MdlN)  # Move the entire Sim folder to recycling bin
                    sprint(f'🟢 - Sim folder {action}d successfully.')
                    i += 1
                except Exception as e:
                    sprint(f'🔴 - failed to {action} Sim folder: {e}')
            else:
                try:  # --- Remove large output files only ---
                    if Pa.imod_V == 'imod5':
                        (Pa.Sim_In / f'{MdlN}.DIS6.grb').unlink(
                            missing_ok=True
                        )  # .grb is usually big and we don't need it.
                        sh.rmtree(Pa.Sim_Out, onexc=_on_rm_error)  # Remove folder containing HD and CBC
                        for item in Pa.MSW.iterdir():  # Remove MSW out folders
                            if item.is_dir():
                                sh.rmtree(item, onexc=_on_rm_error)
                    elif Pa.imod_V == 'imod_python':
                        sim_in_path = Pa.Sim_In
                        if sim_in_path.exists():  # large modflow files
                            for item in sim_in_path.iterdir():
                                if item.suffix in ['.hds', '.cbc', '.grb']:
                                    item.unlink(missing_ok=True)
                        for item in Pa.MSW.iterdir():  # Remove MSW out folders
                            if item.is_dir():
                                sh.rmtree(item, onexc=_on_rm_error)
                    sprint(f'🟢 - Sim folder {action} successfully.')
                    i += 1
                except Exception as e:
                    sprint(f'🔴 - failed to {action} large output files: {e}')
        else:
            sprint(f'🔴 - {Pa_MdlN} does not exist.')

        if MdlN.lower() in DF['MdlN'].str.lower().values:
            if i == 1:
                try:  # --- Change log.csv entry ---
                    DF.loc[DF_match_MdlN(DF, MdlN), 'End Status'] = 'Removed Output'
                    DF.loc[DF_match_MdlN(DF, MdlN), 'Date Removed Output'] = DT.now().strftime('%Y-%m-%d %H:%M')
                    DF.to_csv(Pa_log, index=False)  # Save back to CSV
                    sprint('🟢 - log.csv file updated successfully.')
                    i += 1
                except Exception as e:
                    sprint(f'🔴 - failed to update log.csv file: {e}')

            if i == 2:
                sprint(f'\n🟢🟢🟢 - ALL files were successfully {action}d.')
        else:
            sprint(f'🔴 - {MdlN} not found in log.')
    else:
        sprint('🔴🔴🔴 - Reset cancelled by user (you).')
    sprint(Sep)


def rerun_Sim(MdlN: str, cores=None, DAG: bool = True):
    """
    Reruns the simulation by:
        1. Deleting all files in the MldN folder in the Sim folder.
        2. Clearing log.csv.
        3. Deletes Smk log files for MdlN.
        4. Deletes PoP folder for MdlN.
        5. Runs S_from_B to prepare the simulation files again.
    """

    if cores is None:
        cores = max(
            cpu_count() - 2, 1
        )  # Leave 2 cores free for other tasks. If there aren't enough cores available, set to 1.

    reset_Sim(MdlN)

    DF = r_RunLog()

    if MdlN not in DF['MdlN'].values:
        sprint(f'🔴🔴🔴 - {MdlN} not found in the RunLog. Cannot rerun.')
        return
    else:
        Se_Ln = to_Se(MdlN)  # Get the row for the MdlN

        # Prepare arguments for multiprocessing

        args = [('_', Se_Ln, cores, DAG)]

        # Run models in parallel
        with Pool(processes=cores) as pool:
            results = pool.map(RunSim, args)

        # Print results
        for result in results:
            if len(result) == 2:
                model_id, success = result
                if success:
                    sprint(f'🟢🟢 Model {model_id} completed successfully')
                else:
                    sprint(f'🔴🔴 Model {model_id} failed')
            else:
                model_id, success, error = result
                sprint(f'🔴🔴 Model {model_id} failed: {error}')

    sprint(Sep)


def get_elapsed_time_str(start_time: float) -> str:
    """Returns elapsed time as a formatted string.
    Format: 'd.hh:mm:ss' for days or 'hh:mm:ss' when less than a day"""
    elapsed = DT.now() - start_time
    s = int(elapsed.total_seconds())
    d, h, m, s = s // 86400, (s // 3600) % 24, (s // 60) % 60, s % 60

    if d:
        return f'{d}.{h:02}:{m:02}:{s:02}'
    return f'{h:02}:{m:02}:{s:02}'


def run_cmd(cmd, check=True, capture=False):
    return sp.run(cmd, check=check, capture_output=capture, text=True)


def freeze_pixi_env(MdlN: str):
    """
    Freezes the current Python environment by committing changes to tracked files in the git repository.
    The pixi env freezes everything in pixi.lock. The only package that's not included in pixi.lock (WS_Mdl) can also be restored to a previous state by checking out a specific commit.
    """

    l_Fi_to_track = [
        Pa_WS / i for i in ['pixi.toml', 'pixi.lock', 'code/WS_Mdl']
    ]  # If any of these code files changes, the env needs to be frozen.

    try:
        # Ensure we are in repo root
        Pa_repo = run_cmd(['git', 'rev-parse', '--show-toplevel'], capture=True).stdout.strip()
        sprint(f'Repo root: {Pa_repo}')

        # Check for changes in the relevant files
        diff_cmd = ['git', 'status', '--porcelain'] + l_Fi_to_track
        changes = run_cmd(diff_cmd, capture=True).stdout.strip()

        if not changes:
            sprint('⚪️⚪️⚪️ No changes to tracked env/code files. Nothing to commit.')
            return None, None

        sprint('🟢 Changes detected:\n' + changes)

        # Stage changes
        run_cmd(['git', 'add'] + l_Fi_to_track)
        sprint('🟢 Staged changes.')

        # Commit with timestamp
        now = DT.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f'#auto {MdlN} env snapshot - {now}'
        run_cmd(['git', 'commit', '-m', commit_msg])

        # Get the commit hash of the just-created commit
        commit_hash = run_cmd(['git', 'rev-parse', 'HEAD'], capture=True).stdout.strip()
        sprint(f'🟢 Commit hash: {commit_hash}')

        # Get the tag of the latest commit (if any)
        try:
            tag_result = run_cmd(['git', 'describe', '--tags', '--always', 'HEAD'], capture=True)
            tag = tag_result.stdout.strip()
            sprint(f'🟢 Tag: {tag}')
        except sp.CalledProcessError:
            tag = '-'
            sprint('⚪️ No tag found for this commit. Only the hash will be recorded.')

        sprint(f"🟢🟢🟢 Committed changes with message: '{commit_msg}'")

        return commit_hash, tag

    except sp.CalledProcessError as e:
        print(f'🔴🔴🔴 Error running command: {e}', file=sys.stderr)
        sys.exit(1)
