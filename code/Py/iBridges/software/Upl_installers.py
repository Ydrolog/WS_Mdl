import WS_Mdl.utils as U

print(U.Sep)

l_F = ['software/installers']

print(f'Uploading:\n{"\n".join([f"{i:2}/{len(l_F)} - {j}" for i, j in enumerate(l_F, 1)])}\nfolder(s) to iBridges...\n')

S = U.iB_session()

for f in l_F:
    U.Upl(f, S)  # , overwrite=False)

print(U.Sep)
