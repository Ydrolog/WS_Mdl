import os
import numpy as np
import pandas as pd
import rasterio
import imod
from datetime import datetime as DT
from rasterio.transform import from_bounds
from sklearn.metrics import mean_squared_error
from typing import Optional
# from rasterio.transform import from_origin
## How to import:
# import WS_Mdl as WS
# As long as it is installed (check ../../READ_ME.md for instructions), this should work

path_WS = 'C:/OD/WS_Mdl' #666 Make a function (could be __init__, where the other directories get loaded by default (so that I don't have to do it every time))

def IDF_to_TIF(path_IDF: str, path_TIF: Optional[str] = None, crs="EPSG:28992"):
    """ Converts IDF file to TIF file.
        If path_TIF is not provided, it'll be the same as path_IDF, except for the file type ending.
        crs (coordinate reference system) is set to the Amerfoot crs by default, but can be changed for other projects."""
    A, MtDt = imod.idf.read(path_IDF)

    original_creation_date = DT.fromtimestamp(os.path.getctime(path_IDF)).strftime('%Y-%m-%d %H:%M:%S') # Get OG (IDF) file's date modified.
    conversion_date = DT.now().strftime('%Y-%m-%d %H:%M:%S') # Get current time, to write time of convertion to comment

    nrows, ncols = A.shape

    transform = from_bounds(west=MtDt['xmin'],
                            south=MtDt['ymin'],
                            east=MtDt['xmax'],
                            north=MtDt['ymax'],
                            width=ncols,
                            height=nrows)

    meta = {"driver": "GTiff",
            "height": nrows,
            "width": ncols,
            "count": 1,
            "dtype": str(A.dtype),
            "crs": crs,
            "transform": transform}

    if not path_TIF:
        path_TIF = path_IDF.split('.')[0] + '.tif'

    with rasterio.open(path_TIF, "w", **meta) as dst:
        dst.write(A, 1)  # Write band 1
        dst.update_tags(COMMENT=(f"Converted from IDF on {conversion_date}. "
                                 f"Original file created on {original_creation_date}."))
    print(f"{path_TIF} has been saved (GeoTIFF).")


def INI_to_d(path_INI:str):
    d_INI = {}
    with open(path_INI, 'r', encoding="utf-8") as file:
        for l in file:
            l = l.strip()
            if l and not l.startswith("#"):  # Ignore empty lines and comments
                k, v = l.split("=", 1)  # Split at the first '='
                d_INI[k.strip()] = v.strip()  # Remove extra spaces
    return d_INI



class Vld_Mtc:
    formulas = {"NSE": lambda obs, sim: 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
                "RMSE": lambda obs, sim: np.sqrt(mean_squared_error(obs, sim)),
                "MAE": lambda obs, sim: np.mean(np.abs(obs - sim))}

    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        self.formula = self.formulas.get(name)

    def compute(self, obs, sim):
        if self.formula:
            return self.formula(obs, sim)
        else:
            raise ValueError(f"Formula for {self.name} not found!")
        
def S_from_B(MdlN:str):
    """Copies files from the baseline run, renames them for the scenario run, and opens them in the default file editor. Assumes default WS_Mdl folder structure."""

    # Get metadata for Sc run (input to the function) from the RunLog
    DF = pd.read_excel(os.path.join(WS.path_WS, 'Mng/WS_RunLog.xlsx'), sheet_name='RunLog')    
    S = DF.loc[DF['MdlN']==MdlN].squeeze()
    Mdl, SimN_B, SimN = S[['model alias', 'B SimN', 'SimN']]
    MdlN_B = Mdl + str(SimN_B)

    # Prepare directories
    path_INI_B = os.path.join(WS.path_WS, f'models/{Mdl}/code/Mdl_Prep/Mdl_Prep_{MdlN_B}.ini')
    path_INI_S = path_INI_B.replace(MdlN_B, MdlN)
    path_BAT_B = path_INI_B.replace('.ini', '.bat')
    path_BAT_S = path_BAT_B.replace(MdlN_B, MdlN)
    path_PRJ_B = path_INI_B.replace('.ini', '.PRJ')
    path_PRJ_S = path_PRJ_B.replace(MdlN_B, MdlN)

    # Copy .INI, .bat, .prj and make default (those apply to every Sim) modifications
    if not os.path.exists(path_INI_S): # The MdlN is mentioned twice in the .ini file: link to the .prj file and to the .nam file. It has to be replaced.
        sh.copy2(path_INI_B, path_INI_S)
        with open(path_INI_S, 'r') as f1:
            contents = f1.read()
        with open(path_INI_S, 'w') as f2:
            f2.write(contents.replace(MdlN_B, MdlN))
        os.startfile(path_INI_S) # Then we'll open it to make any other changes we want to make.
    else:
        print(f"{path_INI_S} already exists. If you want it to be replaced, you have to delete it manually before running this command.")

    if not os.path.exists(path_BAT_S): # For the .bat file it's mentioned once at the top. Still it's preferable if it's replaced automatically.
        sh.copy2(path_BAT_B, path_BAT_S)
        with open(path_BAT_S, 'r') as f1:
            contents = f1.read()
        with open(path_BAT_S, 'w') as f2:
            f2.write(contents.replace(MdlN_B, MdlN))
    else:
        print(f"{path_BAT_S} already exists. If you want it to be replaced, you have to delete it manually before running this command.")

    if not os.path.exists(path_PRJ_S): # For the PRJ file, there is no default replacement, so we'll just copy.
        sh.copy2(path_PRJ_B, path_PRJ_S)
        os.startfile(path_PRJ_S) # Then we'll open it to make any other changes we want to make.
    else:
        print(f"{path_PRJ_S} already exists. If you want it to be replaced, you have to delete it manually before running this command.")
