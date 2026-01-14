from ibridges import Session
from WS_Mdl.utils import PJ

Dir_irods = r'C:\Users\Karam014\.irods'

Pw = open(PJ(Dir_irods, 'Pw.txt'), 'r', encoding='utf-8-sig').read().strip()

with Session(irods_env=PJ(Dir_irods, 'irods_environment.json'), password=Pw[::-1]) as S:
    print(S.username)
    #    print(S.default_resc)  # the resource to which data will be uploaded
    print(S.zone)
    print(S.server_version)
    print(S.get_user_info())  # lists user type and groups
    print(S.home)  # default home for iRODS /zone/home/username
