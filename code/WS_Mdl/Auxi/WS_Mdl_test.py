from WS_Mdl import core as C

MdlN = 'NBr52'

M = C.Mdl_N(MdlN)

# -------------------------
# path
# PV = C.MdlN_PaView(MdlN)

# print(PV.as_dict(), '-' * 80)

# print(PV.get('INI'), '-' * 80)

# print(PV.keys(), '-' * 80)
# print(PV.items(), '-' * 80)
# print(PV.values(), '-' * 80)

# -------------------------
# Runtime
# C.timed_import('imod')
# C.timed_execution(C.get_Mdl, MdlN)

# -------------------------
# text.py
print(C.r_Txt_Lns(M.Pa.LST_Sim))
