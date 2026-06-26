# %% Imports
from WS_Mdl.core.mdl import Mdl_N
import pandas as pd

# %%
M = Mdl_N('NBr100')

# %%
DF = pd.read_csv(r"g:\models\NBr\Sim\NBr99\modflow6\imported_model\NBr99.SFR6.obs.output.csv")

# %%
