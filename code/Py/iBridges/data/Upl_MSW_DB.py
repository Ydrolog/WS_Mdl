import WS_Mdl.utils as U

print(U.pre_Sign)
Fo = 'data/MSW_DB.tar.gz'

print(f'Uploading "{Fo}" folder to iBridges...\n')

S = U.iB_session()

U.iB_Upl(Fo, S, l_exceptions=[])  # , overwrite=False)

print(U.post_Sign)
