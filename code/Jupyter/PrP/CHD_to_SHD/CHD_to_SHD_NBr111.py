"""Script to make SHD files from CHD files.
iMOD doesn't accept nan values as SHD, but CHD files have nan (for cells where the layer thickness is 0, thus the model is inactive).
We'll fill them with interpolated values. The values don't really matter.
Some layers are inactive (not due to IBOUND, but cause Thk=0) in large parts of the model area. We'll fill (interpolate between, then extrapolate) the NaN values to the extend of L1 (which is active everywhere). Some layersare completely inactive. Those will be filled with values copied from the layer above.
This script can be improved when I have time. Then I should convert it to a .py file.
"""

"""
Even layers were missing from the folder. I created them just for the SHD file creation, but put them in a Ss folder so they don't cause confusion, e.g. when counting the files.
"""

# %%. Libraries
import imod
import numpy as np
import xarray as xr
from WS_Mdl.core import Mdl_N

# %% Options
MdlN = 'NBr111'
MdlN_CHD = 'NBr111'
date_B = '19991228'
date_S = '20000101'
M = Mdl_N(MdlN)
Pa_CHD = M.Pa.WS / rf'models\NBr\In\CHD\{MdlN_CHD}'
Pa_SHD = M.Pa.WS / rf'models\NBr\In\SHD\{MdlN}'
name = 'LHM_HD'

# %% Read CHD, fill (interpolate), save as SHD
l_CHD = list(Pa_CHD.glob(f'{name}_{date_B}*.idf'))
DA_CHD = imod.formats.idf.open(l_CHD, pattern=f'{{name}}_{date_B}_L{{layer}}_NBr1')

# %% Sort coordinates to allow interpolation
reversed_y = not np.all(np.diff(DA_CHD.y.values) > 0)
reversed_x = not np.all(np.diff(DA_CHD.x.values) > 0)
if reversed_y:
    DA_CHD = DA_CHD.sortby('y')
if reversed_x:
    DA_CHD = DA_CHD.sortby('x')
# fig, ax = plt.subplots()
# img = ax.imshow(mask_valid, cmap="gray")
# cbar = fig.colorbar(img, ax=ax)
mask_valid = ~DA_CHD.isel(layer=0).isnull()  # Get valid mask from layer 0

# %% Fill missing values with inter/extrapolation.
DA_CHD_interp_list = []
for i in range(DA_CHD.sizes['layer']):
    layer_i = DA_CHD.isel(layer=i)

    if layer_i.isnull().all():
        filled = DA_CHD_interp_list[-1]  # if the layer is all NaN use previous L
    else:
        filled = (
            layer_i.interpolate_na(dim='y', method='linear')
            .interpolate_na(dim='x', method='linear')
            .fillna(layer_i.ffill('y').bfill('y').ffill('x').bfill('x'))
            .where(mask_valid)
        )
    DA_CHD_interp_list.append(filled.assign_coords(layer=DA_CHD.layer.isel(layer=i)))

DA_CHD_interp = xr.concat(DA_CHD_interp_list, dim='layer')
DA_CHD_interp['layer'] = DA_CHD['layer']

# %% Reverse coords back to original orientation
if reversed_y:
    DA_CHD_interp = DA_CHD_interp.sortby('y', ascending=False)
if reversed_x:
    DA_CHD_interp = DA_CHD_interp.sortby('x', ascending=False)

# %% Expand dimensions and save
DA_CHD_interp = DA_CHD_interp.expand_dims(name=[f'SHD_{date_S}'])
imod.idf.save(Pa_SHD / 'dummy.idf', DA_CHD_interp, pattern=f'{{name}}_L{{layer}}_{MdlN}.IDF')

# %% Write SHD block
for i in range(37):
    print(
        rf" 1,2, {i + 1:03},   1.000000    ,   0.000000    ,  -999.9900    , '..\..\In\SHD\{MdlN}\SHD_{date_S}_L{i + 1}_{MdlN}.IDF' >>> (shd) starting heads (idf) <<<"
    )

# %% Write metadata file in the same folder
with open(Pa_SHD / '_metadata.txt    ', 'w') as f:
    f.write(
        rf"This file was produced by 'G:\code\PrP\CHD_to_SHD\CHD_to_SHD_{MdlN}.py' cause significant differences were spotted between the CHD (of the 1st SP) and SHDs of NBr38.)"
    )
