import pandas as pd
from WS_Mdl.core.mdl import Mdl_N


def to_DF(MdlN):
    M = Mdl_N(MdlN)

    with open(M.Pa.MSW / 'para_sim.inp', 'r') as f:
        lines = [i for i in f.read().split('\n') if i != '' and not i.startswith('*')]

        data = []
        for line in lines:
            comment = ''
            if '!' in line:
                line_part, comment_part = line.split('!', 1)
                comment = comment_part.strip()
                line = line_part

            if '=' in line:
                param, val = line.split('=', 1)
                data.append([param.strip(), val.strip(), comment])

        DF = pd.DataFrame(data, columns=['Parameter', 'Value', 'Comment'])

    return DF
