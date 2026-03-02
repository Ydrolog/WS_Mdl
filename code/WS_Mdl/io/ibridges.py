import os
import tarfile
from pathlib import Path

from ibridges import IrodsPath as iPa
from ibridges import Session
from ibridges import download as Dl
from ibridges import upload as Upl
from tqdm import tqdm
from WS_Mdl.core.style import Sep_2, bold, sprint, style_reset, warn


def l_Fis_Exc(Pa: Path | str, l_exceptions=['.7z', '.aux', '.xml']):
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
    sprint(Sep_2, indent=1)
    sprint(
        f'{len(l_)} files in {Pa.name} excluding exceptions:',
        *[f'{i} {j.name}' for i, j in enumerate(l_, 1)],
        sep='\n',
        end='\n',
    )
    sprint(Sep_2, indent=1)
    return l_


def iB_get_Pw(Dir_irods=rf'C:\Users\{os.getlogin()}\.irods', Pw_txt: str = 'Pw.txt', inverse: bool = True):
    """Reads iRODS password from Pw.txt file."""
    Pw = open(Path(Dir_irods) / Pw_txt, 'r', encoding='utf-8-sig').read().strip()  # Read password from Pw.txt.
    return Pw[::-1] if inverse else Pw


class iB_session(Session):
    def __init__(self, Dir_irods=rf'C:\Users\{os.getlogin()}\.irods', PW_txt: str = 'Pw.txt'):
        """Loads an iBridges iRODS session using the irods_environment.json file and password from PW_txt."""
        dir_irods = Path(Dir_irods)
        Pw = iB_get_Pw(Dir_irods, Pw_txt=PW_txt)
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


def iB_Upl(
    F: str,
    S,
    on_error='warn',
    l_exceptions=['.dvc', '.7z', '.aux', '.xml'],
    overwrite=False,
    subdir='research-ws-imod',
):
    """Uploads an iBridges file/folder."""

    CWD = iPa(S, '~') / subdir
    Pa_Loc = Path(f'G:/{F}/')
    l_Fi_data = l_Fis_Exc(Pa_Loc, l_exceptions=l_exceptions)

    print(f'Uploading from: {Pa_Loc}')
    if Pa_Loc.is_file():
        if l_Fi_data:
            Target = CWD / F
            print(f'Uploading to:   {Target}')
            if not Target.parent.exists():
                Target.parent.create_collection()
            print('1/1', Target)
            Upl(l_Fi_data[0], Target, on_error=on_error)
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
            Upl(Pa, Target, on_error=on_error, overwrite=overwrite)
            sprint(Sep_2, indent=1)


def iB_Dl(F: str, S, on_error='warn', overwrite=False, subdir='research-ws-imod', decompress: bool = True):
    """Downloads an iBridges file/folder."""

    Pa_Rmt = iPa(S, '~') / subdir / F
    Pa_Loc = Path(f'G:/{F}')

    if Pa_Rmt.dataobject_exists():
        if not Pa_Loc.parent.exists():
            Pa_Loc.parent.mkdir(parents=True, exist_ok=True)
        print('1/1', Pa_Loc)
        Dl(Pa_Rmt, Pa_Loc, overwrite=overwrite, on_error=on_error)

    elif Pa_Rmt.collection_exists():
        Dest = Pa_Loc.parent
        if not Dest.exists():
            Dest.mkdir(parents=True, exist_ok=True)

        print(f'Downloading folder: {Pa_Rmt} -> {Pa_Loc}')
        Dl(Pa_Rmt, Dest, overwrite=overwrite, on_error=on_error)
    else:
        sprint(f'{warn}Remote path not found: {Pa_Rmt}')
        return

    # Post-process: Decompress .tar.gz files
    if decompress and Pa_Loc.exists():

        def decompress_and_clean(file_path):  # Helper to decompress and remove .tar.gz files
            if str(file_path).endswith('.tar.gz'):
                print(f'Decompressing {file_path}...')
                try:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        members = tar.getmembers()
                        pbar = tqdm(total=len(members), desc=f'Extracting {file_path.name}', unit='file')
                        for member in members:
                            tar.extract(member, path=file_path.parent)
                            pbar.update(1)
                        pbar.close()
                    os.remove(file_path)
                except Exception as e:
                    print(f'{warn}Failed to decompress {file_path}: {e}')

        if Pa_Loc.is_file():
            decompress_and_clean(Pa_Loc)
        elif Pa_Loc.is_dir():
            for root, _, files in Pa_Loc.walk():
                for file in files:
                    decompress_and_clean(Path(root) / file)
    sprint(Sep_2, indent=1)
