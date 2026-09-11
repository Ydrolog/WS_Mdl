import os
import tarfile
import warnings
from pathlib import Path

from ibridges import IrodsPath as iPa
from ibridges import Session, download, upload
from tqdm import tqdm
from WS_Mdl.core.defaults import Pa_WS
from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import VERBOSE, Sep, Sep_2, blue, bold, green, sprint, style_reset, warn

__all__ = ['Dl', 'Dl_MdlN_PoP_Out', 'get_Pw']


def l_Fis_Exc(Pa: Path | str, l_exceptions=['.7z', '.aux', '.xml'], verbose: bool = True):
    Pa = Path(Pa)
    l_ = []
    if Pa.is_file():
        if Pa.name not in l_exceptions and Pa.suffix not in l_exceptions:
            l_ = [Pa]
    else:
        for root, dirs, files in Pa.walk():
            dirs[:] = [d for d in dirs if d not in l_exceptions]
            for f in files:
                if f not in l_exceptions and Path(f).suffix not in l_exceptions:
                    l_.append(Path(root) / f)
    sprint(Sep_2, indent=1, verbose_in=verbose)
    sprint(
        f'{len(l_)} files in {Pa.name} excluding exceptions:',
        *[f'{i} {j.name}' for i, j in enumerate(l_, 1)],
        sep='\n',
        end='\n',
        style=blue,
    )
    sprint(Sep_2, indent=1, verbose_out=VERBOSE)
    return l_


def get_Pw(Dir_irods=rf'C:\Users\{os.getlogin()}\.irods', Pw_txt: str = 'Pw.txt', inverse: bool = True):
    """Reads iRODS/PAM password from Pw.txt.

    Note: by default this assumes the password is stored reversed in the text file (a light obfuscation).
    If your Pw.txt contains the real password as-is, pass inverse=False.
    """
    pw_path = Path(Dir_irods) / Pw_txt
    if not pw_path.exists():
        raise FileNotFoundError(f'Password file not found: {pw_path}')
    pw = pw_path.read_text(encoding='utf-8-sig').strip()
    return pw[::-1] if inverse else pw


class iB_session(Session):
    def __init__(
        self,
        Dir_irods=rf'C:\Users\{os.getlogin()}\.irods',
        PW_txt: str = 'Pw.txt',
        inverse: bool | None = None,
    ):
        """Loads an iBridges iRODS session.

        Args:
            Dir_irods: Folder containing `irods_environment.json`.
            PW_txt: Password file name in Dir_irods.
            inverse: Whether to reverse the read password (defaults to env var override if provided).
                Env vars: `IBRIDGES_PW_INVERSE` or `IB_PW_INVERSE` (true/false, 1/0).
        """
        dir_irods = Path(Dir_irods)
        Pw = get_Pw(Dir_irods, Pw_txt=PW_txt)
        super().__init__(irods_env=dir_irods / 'irods_environment.json', password=Pw)

    def info(self):
        """Prints iBridges session info."""
        sprint(Sep_2, indent=1)
        sprint(f'{bold}iBridges session info:{style_reset}')
        sprint(f'{"username":15}:', self.username)
        sprint(f'{"zone":15}:', self.zone)
        sprint(f'{"server_version":15}:', self.server_version)
        sprint(f'{"user_info":15}:', self.get_user_info())  # lists user type and groups
        sprint(f'{"home":15}:', self.home)  # default home for iRODS /zone/home/username
        sprint(Sep_2, indent=1)


def Upl(
    F: str,
    S: iB_session,
    on_error='warn',
    l_exceptions=['.dvc', '.7z', '.aux', '.xml'],
    overwrite=False,
    subdir='research-ws-imod',
    Pa_base: str | Path = Pa_WS,
):
    """Uploads an iBridges file/folder."""

    CWD = iPa(S, '~') / subdir
    Pa_Loc = Path(Pa_base) / Path(F)
    l_Fi_data = l_Fis_Exc(Pa_Loc, l_exceptions=l_exceptions)

    print(f'Uploading from: {Pa_Loc}')
    if Pa_Loc.is_file():
        if l_Fi_data:
            Target = CWD / Pa_Loc.relative_to(Pa_base)
            print(f'Uploading to:   {Target}')
            if not Target.parent.exists():
                Target.parent.create_collection()
            print('1/1', Target)
            upload(l_Fi_data[0], Target, on_error=on_error)
            sprint(Sep_2, indent=1)
    else:
        CWD_Fo = CWD / F
        print(f'Uploading to:   {CWD_Fo}')
        if not CWD_Fo.exists():
            CWD_Fo.create_collection()
        for i, Pa in enumerate(l_Fi_data, 1):
            Rel = Pa.relative_to(Pa_Loc)
            Target = CWD_Fo / Rel
            if not Target.parent.exists():
                Target.parent.create_collection()
            print(f'{i}/{len(l_Fi_data)}', Target)
            upload(Pa, Target, on_error=on_error, overwrite=overwrite)
            sprint(Sep_2, indent=1)


def _extract_archive(file_path, overwrite=False, on_error='warn'):
    """Retain successful archives; only mark extraction complete after all members finish."""
    file_path = Path(file_path)
    marker = file_path.with_name(file_path.name + '.extracted')
    signature = f'{file_path.stat().st_size}:{file_path.stat().st_mtime_ns}'
    if not overwrite and marker.is_file() and marker.read_text() == signature:
        print(f'Skipping already extracted archive: {file_path}')
        return

    print(f'Decompressing {file_path}...')
    skipped = 0
    try:
        marker.unlink(missing_ok=True)
        with tarfile.open(file_path, 'r:gz') as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc=f'Extracting {file_path.name}', unit='file') as pbar:
                for member in members:
                    # Validate even skipped members before resolving their destination.
                    safe_member = tarfile.data_filter(member, str(file_path.parent))
                    if safe_member is None:
                        raise ValueError(f'Archive member rejected: {member.name}')
                    target = file_path.parent / safe_member.name
                    if not overwrite and target.exists():
                        if not ((member.isdir() and target.is_dir()) or
                                (member.isfile() and target.is_file() and not target.is_symlink()
                                 and target.stat().st_size == member.size)):
                            raise FileExistsError(f'Existing archive destination conflicts: {target}')
                        skipped += 1
                    else:
                        existed = target.exists()
                        try:
                            tar.extract(member, path=file_path.parent, filter='data')
                        except Exception:
                            # Do not leave a new, incomplete file that blocks a retry.
                            if not existed and member.isfile() and target.is_file():
                                target.unlink()
                            raise
                    pbar.update(1)
        marker.write_text(signature)
        print(f'Extraction complete: {file_path} ({skipped} existing entries skipped)')
    except Exception as exc:
        cleanup_errors = []
        for path in (marker, file_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f'Could not remove {path}: {cleanup_exc}')
        message = f'Failed to decompress {file_path}: {exc}. '
        message += (' '.join(cleanup_errors) if cleanup_errors else
                    'Archive removed so the next run can download it again.')
        if on_error == 'raise':
            raise RuntimeError(message) from exc
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def Dl(F: str, S: iB_session, on_error='warn', overwrite=False, subdir='research-ws-imod', decompress: bool = True):
    """
    Downloads an iBridges file/folder, e.g.: Dl('models/NBr/code/snakemake/log', S, overwrite=True)

    Successful archives are retained with a .extracted completion marker. With
    overwrite=False, existing downloads and completed extractions are skipped.
    Failed extraction removes the archive for retry on the next download.
    """

    Pa_Rmt = iPa(S, '~') / subdir / F
    Pa_Loc = Path(f'G:/{F}')

    if Pa_Rmt.dataobject_exists():
        if not Pa_Loc.parent.exists():
            Pa_Loc.parent.mkdir(parents=True, exist_ok=True)
        print('1/1', Pa_Loc)
        if overwrite or not Pa_Loc.exists():
            download(Pa_Rmt, Pa_Loc, overwrite=overwrite, on_error=on_error)
        else:
            print(f'Skipping existing file: {Pa_Loc}')

    elif Pa_Rmt.collection_exists():
        Dest = Pa_Loc.parent
        if not Dest.exists():
            Dest.mkdir(parents=True, exist_ok=True)

        sprint(f'Downloading folder:\n{Pa_Rmt} -> {Pa_Loc}')
        download(Pa_Rmt, Dest, overwrite=overwrite, on_error=on_error)
    else:
        sprint(f'{warn}Remote path not found: {Pa_Rmt}')
        return

    # Post-process: Decompress .tar.gz files
    if decompress and Pa_Loc.exists():

        def decompress_and_clean(file_path):
            if str(file_path).endswith('.tar.gz'):
                _extract_archive(file_path, overwrite=overwrite, on_error=on_error)

        if Pa_Loc.is_file():
            decompress_and_clean(Pa_Loc)
        elif Pa_Loc.is_dir():
            for root, _, files in Pa_Loc.walk():
                for file in files:
                    decompress_and_clean(Path(root) / file)
    sprint(Sep_2, indent=1)


def Upl_Sim_Out(MdlN):
    """
    To be used after RunMng has finished running a Smk file. Uploads only PoPed Out and Smk related files (log, temp files, DAG)
    """

    sprint(Sep)
    sprint(f'--- Upl_Sim_Out({MdlN}) ... ', style=green)
    S = iB_session()

    M = Mdl_N(MdlN)

    # MF6 important Ins #666 finish up
    l_Fi = []
    for F in l_Fi:
        Upl(f'models/{M.alias}/Sim/{MdlN}/modflow6/{F}', S, overwrite=True)

    # PoP files
    sprint(' -- Uploading PoP Out files ...', set_time=True, end='')
    Upl(f'models/{M.alias}/PoP/Out/{MdlN}', S, overwrite=True)  # PoP Out (most important)
    Upl(f'models/{M.alias}/PoP/In', S, overwrite=True)  # PoP Out (most important)
    sprint('🟢', print_time=True)

    # Smk files
    sprint(' -- Uploading Smk files ...', set_time=True, end='')
    Upl(f'models/{M.alias}/code/snakemake/log', S, overwrite=True)
    Upl(f'models/{M.alias}/code/snakemake/DAG', S, overwrite=True)
    Upl(f'models/{M.alias}/code/snakemake/temp', S, overwrite=True)
    sprint('🟢', print_time=True)

    sprint(Sep)


def Dl_Sim_Out(MdlN):
    """
    Downloads PoPed Out and Smk related files (log, temp files, DAG) for a given MdlN.
    """

    sprint(Sep)
    sprint(f'--- Dl_MdlN_PoP_Out({MdlN}) ... ', style=green)

    S = iB_session()
    M = Mdl_N(MdlN)

    # PoP files
    sprint(' -- Downloading PoP Out files ...', set_time=True)
    Dl(f'models/{M.alias}/PoP/Out/{MdlN}', S, overwrite=True)
    Dl(f'models/{M.alias}/PoP/In', S, overwrite=True)
    sprint(' -- 🟢', print_time=True)

    # Smk files
    sprint(' -- Downloading Smk files ...', set_time=True)
    Dl(f'models/{M.alias}/code/snakemake/log', S, overwrite=True)
    Dl(f'models/{M.alias}/code/snakemake/DAG', S, overwrite=True)
    Dl(f'models/{M.alias}/code/snakemake/temp', S, overwrite=True)
    sprint(' -- 🟢', print_time=True)

    sprint(Sep)
