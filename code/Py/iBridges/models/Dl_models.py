import WS_Mdl.utils as U

print(U.Sep)

l_F = ['models']

print(f'Downloading "{l_F}" to from iBridges...\n')

S = U.iB_session()

S.info()

for F in l_F:
    U.Dl(F, S)  # , overwrite=False)

print(U.Sep)
