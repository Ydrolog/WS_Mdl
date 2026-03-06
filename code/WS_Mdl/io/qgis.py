import os
import re
import shutil as sh
import sys
import xml.etree.ElementTree as ET
import zipfile as ZF
from os import listdir as LD
from os import makedirs as MDs
from os.path import basename as PBN
from os.path import dirname as PDN
from os.path import join as PJ

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint

sys.excepthook = sys.__excepthook__


def update_MM(MdlN, MdlN_MM_B=None):
    """Updates the MM (QGIS projct containing model data)."""

    sprint(Sep)
    sprint(f' *****   Creating MM for {MdlN}   ***** ')

    M = Mdl_N(MdlN)
    d_Pa = M.Pa
    Pa_QGZ, Pa_QGZ_B = d_Pa['MM'], d_Pa['MM_B']
    Mdl = M.alias

    if MdlN_MM_B is not None:  # Replace MdlN_B with another MdlN if requested.
        Pa_QGZ_B = Pa_QGZ_B.replace(d_Pa['MdlN_B'], MdlN_MM_B)

    MDs(PBN(Pa_QGZ), exist_ok=True)  # Ensure destination folder exists
    os.makedirs(PDN(Pa_QGZ), exist_ok=True)
    sh.copy(Pa_QGZ_B, Pa_QGZ)  # Copy the QGIS file
    sprint(f'Copied QGIS project from {Pa_QGZ_B} to {Pa_QGZ}.\nUpdating layer path ...')

    Pa_temp = PJ(PDN(Pa_QGZ), 'temp')  # Path to temporarily extract QGZ contents
    MDs(Pa_temp, exist_ok=True)

    with ZF.ZipFile(Pa_QGZ_B, 'r') as zip_ref:  # Unzip .qgz
        zip_ref.extractall(Pa_temp)

    Pa_QGS = PJ(
        Pa_temp, LD(Pa_temp)[0]
    )  # Path to the unzipped QGIS project file. This used to be: Pa_QGS = PJ(Pa_temp, PBN(Pa_QGZ).replace('.qgz', '.qgs')), but the extracted file name may vary.
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
                sprint(f'🔴 ERROR: multiple non-identical {Mdl}Ns found in path: {matches}')
                sys.exit('Fix the path containing non-identical MdlNs, then re-run me.')
            else:
                MdlX = f'{Mdl}{matches[0]}'

                Pa_full = os.path.normpath(PJ(PDN(Pa_QGZ), path.replace(MdlX, MdlN)))
                if (MdlX != MdlN) and (os.path.exists(Pa_full)):
                    Pa_X = path.replace(MdlX, MdlN)
                    DS.text = f'{Pa_X}|{suffix}' if suffix else Pa_X
                    sprint(f' - 🟢 Updated {MdlX} → {MdlN} in {Pa_full}')
                # else:
                # sprint(" - OK (no change)")
        # else:
        #     sprint(" - No Mdl in path")
        # sprint('-'*50)

    tree.write(Pa_QGS, encoding='utf-8', xml_declaration=True)  # Save the modified .qgs file

    with ZF.ZipFile(Pa_QGZ, 'w', ZF.ZIP_DEFLATED) as zipf:  # Zip back into .qgz
        for foldername, _, filenames in os.walk(Pa_temp):
            for filename in filenames:
                filepath = PJ(foldername, filename)
                arcname = os.path.relpath(filepath, Pa_temp)
                zipf.write(filepath, arcname)

    sh.rmtree(Pa_temp)  # Remove the temporary folder
    sprint(f'\n🟢🟢🟢 | MM for {MdlN} has been updated.')
    sprint(Sep)
