# %% Imports
import calendar
from pathlib import Path

# %% Options
year_start = 2036
year_end = 2065
value = 7
MdlN = 'NBr122'  # simulatienummer, bv. 'NBr120'

out_dir = Path('../../../../models/NBr/In/CAP/tiop_sim') / MdlN
out_path = out_dir / 'tiop_sim.inp'

# %% Write file
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', newline='\n') as f:
    for year in range(year_start, year_end + 1):
        n_days = 366 if calendar.isleap(year) else 365
        for day in range(n_days):
            f.write(f"{day:15.2f}  {year:4d}{value:6d}\n")

# %%