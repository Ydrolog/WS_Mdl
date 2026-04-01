# import os
import re
import shutil as sh
import sys
import xml.etree.ElementTree as ET
import zipfile as ZF
from pathlib import Path

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint

sys.excepthook = sys.__excepthook__


def update_MM(MdlN, MdlN_MM_B=None):
    """Updates the MM (QGIS projct containing model data)."""

    sprint(Sep)
    sprint(f' *****   Creating MM for {MdlN}   ***** ')

    M = Mdl_N(
        MdlN,
    )
    Pa_QGZ, Pa_QGZ_B = M.Pa.MM, M.Pa_B.MM
    Mdl = M.alias

    if MdlN_MM_B is not None:  # Replace MdlN_B with another MdlN if requested.
        Pa_QGZ_B = Path(str(Pa_QGZ_B).replace(M.B, MdlN_MM_B))

    Pa_QGZ.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination folder exists
    print(Pa_QGZ_B, Pa_QGZ)
    sh.copy(Pa_QGZ_B, Pa_QGZ)  # Copy the QGIS file
    sprint(f'Copied QGIS project from {Pa_QGZ_B} to {Pa_QGZ}.\nUpdating layer path ...')

    Pa_temp = Pa_QGZ.parent / 'temp'  # Path to temporarily extract QGZ contents
    Pa_temp.mkdir(parents=True, exist_ok=True)

    with ZF.ZipFile(Pa_QGZ_B, 'r') as zip_ref:  # Unzip .qgz
        zip_ref.extractall(Pa_temp)

    Pa_QGS = Pa_temp / Pa_QGZ_B.name.replace('.qgz', '.qgs')
    # PJ(
    #     Pa_temp, LD(Pa_temp)[0]
    # )  # Path to the unzipped QGIS project file. This used to be: Pa_QGS = PJ(Pa_temp, PBN(Pa_QGZ).replace('.qgz', '.qgs')), but the extracted file name may vary.
    tree = ET.parse(Pa_QGS)
    root = tree.getroot()

    # Update datasource paths
    for i, DS in enumerate(root.iter('datasource')):
        DS_text = DS.text
        # sprint(i, DS_text)

        if not DS_text:
            # sprint(' - X - Not text')
            # sprint('-'*50)
            continue

        if '|' in DS_text:
            path, suffix = DS_text.split('|', 1)
        else:
            path, suffix = DS_text, ''

        if Mdl in path:
            matches = re.findall(rf'{re.escape(Mdl)}(\d+)', path)
            if len(set(matches)) > 1:
                sprint(f'🔴 ERROR: multiple non-identical {Mdl}Ns found in path: {path}\nmatches: {matches}')
                # sys.exit('Fix the path containing non-identical MdlNs, then re-run me.')
                continue
            else:
                MdlX = f'{Mdl}{matches[0]}'

                Pa_full = (Pa_QGZ.parent / path.replace(MdlX, MdlN)).absolute()
                if (MdlX != MdlN) and (Pa_full.exists()):
                    Pa_X = path.replace(MdlX, MdlN)
                    DS.text = f'{Pa_X}|{suffix}' if suffix else Pa_X
                    sprint(f'  - 🟢 Updated {MdlX} → {MdlN} in {Pa_full}')
                # else:
                # sprint(" - OK (no change)")
        # else:
        #     sprint(" - No Mdl in path")
        # sprint('-'*50)

    tree.write(Pa_QGS, encoding='utf-8', xml_declaration=True)  # Save the modified .qgs file

    with ZF.ZipFile(Pa_QGZ, 'w', ZF.ZIP_DEFLATED) as zipf:  # Zip back into .qgz
        # Mirror the old os.walk behavior: include every file under temp with relative arcname.
        for filepath in Pa_temp.rglob('*'):
            if filepath.is_file():
                arcname = filepath.relative_to(Pa_temp)
                zipf.write(filepath, arcname)

    sh.rmtree(Pa_temp)  # Remove the temporary folder
    sprint(f'\n🟢🟢🟢 | MM for {MdlN} has been updated.')
    sprint(Sep)
