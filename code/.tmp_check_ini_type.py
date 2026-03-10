from WS_Mdl.core.mdl import Mdl_N
M = Mdl_N('NBr50')
print('type(M.Pa.INI)=', type(M.Pa.INI), M.Pa.INI)
ini = M.INI
print('type(M.INI)=', type(ini))
print('has SDATE=', 'SDATE' in ini, '; attr SDATE=', ini.SDATE)
