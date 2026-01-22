import WS_Mdl.utils as U

print(U.pre_Sign)

l_F = ['software/iMOD5']

print(f'Uploading:\n{"\n".join([f"{i:2}/{len(l_F)} - {j}" for i, j in enumerate(l_F, 1)])}\nfolder(s) to iBridges...\n')

S = U.iB_Session()

for f in l_F:
    U.iB_Upl(f, S)  # , overwrite=False)

print(U.post_Sign)
