import re
import shutil as sh
import sys
import xml.etree.ElementTree as ET
import zipfile as ZF

from WS_Mdl.core.mdl import Mdl_N
from WS_Mdl.core.style import Sep, sprint
from WS_Mdl.core.text import replace_MdlN

sys.excepthook = sys.__excepthook__

__all__ = ['Up_MM']


def _Up_L_references(root, maplayer, replace_reference, datasource):
    """Update the names and cached datasource of one layer throughout the QGIS XML."""

    layer_id = maplayer.findtext('id')
    layer_name = maplayer.find('layername')
    if layer_name is not None and layer_name.text:
        layer_name.text = replace_reference(layer_name.text)

    if not layer_id:
        return

    for tree_layer in root.iter('layer-tree-layer'):
        if tree_layer.get('id') == layer_id:
            if tree_layer.get('name'):
                tree_layer.set('name', replace_reference(tree_layer.get('name')))
            # QGIS stores a second copy of the datasource in the layer tree.
            tree_layer.set('source', datasource)

    # Older QGIS project sections can also cache the display name.
    for legend_layer in root.iter('legendlayer'):
        if any(layer_file.get('layerid') == layer_id for layer_file in legend_layer.iter('legendlayerfile')):
            if legend_layer.get('name'):
                legend_layer.set('name', replace_reference(legend_layer.get('name')))


def Up_MM(MdlN, MdlN_B=None, MdlN_MM_B=None):
    """
    Copy an MM project and update its resolvable layer references.

    MdlN_B: MdlN for comparison layer names
    MdlN_MM_B: MdlN for the QGIS project to copy and defaults to ``MdlN_B``
    The only case where you need MdlN_MM_B is when you want to copy the QGIS project from a MdlN, but compare it to another MdlN.
    """

    sprint(Sep)
    sprint(f' *****   Creating MM for {MdlN}   ***** ')

    M = Mdl_N(MdlN)
    MdlN_B = MdlN_B or M.B
    MdlN_MM_B = MdlN_MM_B or MdlN_B

    M_B = Mdl_N(MdlN_B)
    M_MM_B = Mdl_N(MdlN_MM_B)
    comparison_pattern = re.compile(rf'{re.escape(MdlN_MM_B)}m(?:{re.escape(M_MM_B.alias)})?\d+(?!\d)')
    comparison_replacement = f'{MdlN}m{M_B.N}'

    def replace_reference(value):
        value = comparison_pattern.sub(lambda _: comparison_replacement, value)
        return replace_MdlN(value, MdlN_MM_B, MdlN)

    M.Pa.MM.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination folder exists
    print(M_MM_B.Pa.MM, M.Pa.MM)
    sh.copy(M_MM_B.Pa.MM, M.Pa.MM)  # Copy the QGIS file
    sprint(f'Copied QGIS project from {M_MM_B.Pa.MM} to {M.Pa.MM}.\nUpdating layer path ...')

    Pa_temp = M.Pa.MM.parent / 'temp'  # Path to temporarily extract QGZ contents
    Pa_temp.mkdir(parents=True, exist_ok=True)

    with ZF.ZipFile(M_MM_B.Pa.MM, 'r') as zip_ref:  # Unzip .qgz
        zip_ref.extractall(Pa_temp)

    Pa_QGS = Pa_temp / M_MM_B.Pa.MM.name.replace('.qgz', '.qgs')
    # PJ(
    #     Pa_temp, LD(Pa_temp)[0]
    # )  # Path to the unzipped QGIS project file. This used to be: Pa_QGS = PJ(Pa_temp, PBN(M.Pa.MM).replace('.qgz', '.qgs')), but the extracted file name may vary.
    tree = ET.parse(Pa_QGS)
    root = tree.getroot()

    parent_by_child = {child: parent for parent in root.iter() for child in parent}

    # Update datasource paths, suffixes, and linked QGIS layer names.
    for DS in root.iter('datasource'):
        DS_text = DS.text

        if not DS_text:
            continue

        if '|' in DS_text:
            path, suffix = DS_text.split('|', 1)
        else:
            path, suffix = DS_text, ''

        Pa_X = replace_reference(path)
        suffix_X = replace_reference(suffix)
        if (Pa_X, suffix_X) == (path, suffix):
            continue

        Pa_full = (M.Pa.MM.parent / Pa_X).absolute()
        if Pa_full.exists():
            DS.text = f'{Pa_X}|{suffix_X}' if suffix_X else Pa_X
            maplayer = parent_by_child.get(DS)
            if maplayer is not None and maplayer.tag == 'maplayer':
                _Up_L_references(root, maplayer, replace_reference, DS.text)
            sprint(f'  - 🟢 Updated {MdlN_MM_B} → {MdlN} in {Pa_full}')

    tree.write(Pa_QGS, encoding='utf-8', xml_declaration=True)  # Save the modified .qgs file

    with ZF.ZipFile(M.Pa.MM, 'w', ZF.ZIP_DEFLATED) as zipf:  # Zip back into .qgz
        # Mirror the old os.walk behavior: include every file under temp with relative arcname.
        for filepath in Pa_temp.rglob('*'):
            if filepath.is_file():
                arcname = filepath.relative_to(Pa_temp)
                zipf.write(filepath, arcname)

    sh.rmtree(Pa_temp)  # Remove the temporary folder
    sprint(f'\n🟢🟢🟢 | MM for {MdlN} has been updated.')
    sprint(Sep)
