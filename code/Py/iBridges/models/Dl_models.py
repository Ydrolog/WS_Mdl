import WS_Mdl.utils as U

print(U.pre_Sign)

l_F = ['models']

print(f'Downloading "{l_F}" to from iBridges...\n')

S = U.iB_Session()

S.info()

for F in l_F:
    U.iB_Dl(F, S)  # , overwrite=False)

print(U.post_Sign)
