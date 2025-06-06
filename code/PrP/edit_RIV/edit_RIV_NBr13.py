import WS_Mdl as WS
import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import imod

MdlN, MdlN_B = 'NBr13', 'NBr1'

Stg_increase_m = 0.2

d_paths_B = WS.get_MdlN_paths(MdlN_B) # Get default directories
Dir_RIV = PJ(d_paths_B['path_Mdl'], 'In/RIV')

d_paths_S = WS.get_MdlN_paths((MdlN))

# Read B RIV files
l_RIV_B = [f for f in LD(Dir_RIV) if (MdlN_B in f) and ('RIV_Stg' in f) and ('.IDF' in f.upper()) and ('.dvc' not in f.lower())]

# Edit and save S RIV files
for f in l_RIV_B[:]:
    IDF = imod.formats.idf.open(PJ(Dir_RIV, f))  # Load IDF
    IDF = IDF.compute() if IDF.chunks is not None else IDF  # Ensure it's computed if it's a Dask array
    IDF_modified = IDF + Stg_increase_m
    Dir_Out = PJ(Dir_RIV, MdlN, f.replace(MdlN_B, MdlN))  # Define output file name
    MDs(PDN(Dir_Out), exist_ok=True) # ensures the output directory exists
    imod.formats.idf.save(Dir_Out, IDF_modified)  # Save the modified IDF