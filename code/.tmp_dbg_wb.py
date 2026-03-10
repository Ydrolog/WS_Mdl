import traceback
import WS_Mdl.imod.pop.wb as wb
from WS_Mdl.imod.pop.wb import Diff_to_xlsx as D
print('set_verbose=', wb.set_verbose, type(wb.set_verbose))
wb.set_verbose(True)
print('set_verbose ok')
try:
    D('NBr50', 'NBr43', '1994-01-01')
except Exception:
    traceback.print_exc()
