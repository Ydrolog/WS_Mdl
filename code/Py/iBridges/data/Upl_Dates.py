import WS_Mdl.utils as U

print(U.pre_Sign)
l_F = ['data/Dates.csv', 'data/.gitignore']

print(f'Uploading "{l_F}" folder to iBridges...\n')

S = U.iB_session()

for F in l_F:
    U.iB_Upl(F, S)  # , overwrite=False)

print(U.post_Sign)
