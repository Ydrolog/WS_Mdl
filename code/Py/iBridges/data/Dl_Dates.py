import WS_Mdl.utils as U

print(U.pre_Sign)
F = 'data/test.txt'

print(f'Uploading "{F}" folder to iBridges...\n')

S = U.iB_load_session()

U.iB_Dl(F, S)  # , overwrite=False)

print(U.post_Sign)
