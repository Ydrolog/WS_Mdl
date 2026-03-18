import flopy as fp
import pandas as pd
from WS_Mdl.core.mdl import Mdl_N


def Diff_to_xlsx(
    MdlN: str, MdlN_B: str = None, date: str = None, cumulative=True, sum_Pkg: bool = True, Pa_Out: str | None = None
):  # 666 add option to do cumulative of just for that SP
    """
    Compares the water budget of two models (MdlN and MdlN_B) for a specific date and saves the differences to an Excel file.
    - date: YYYY-MM-DD
    - sum_Pkg: if True, sum rows where the first 3 index characters match
    """

    # Load basics
    M = Mdl_N(MdlN)
    M_B = Mdl_N(MdlN_B) if MdlN_B else Mdl_N(M.B)
    start_date = pd.to_datetime(str(M.INI.SDATE), format='%Y%m%d')

    # Load budget to dataframes. fp.utils.Mf6ListBudget returns a tuple. 1st item is WB for each SP. 2nd item is cumulative.
    i = 1 if cumulative else 0
    DF_1 = fp.utils.Mf6ListBudget(M.Pa.LST_Mdl).get_dataframes()[i]
    DF_2 = fp.utils.Mf6ListBudget(M_B.Pa.LST_Mdl).get_dataframes()[i]

    DF_1.index = pd.date_range(start=start_date, periods=len(DF_1), freq='D')
    DF_2.index = pd.date_range(start=start_date, periods=len(DF_2), freq='D')
    print(DF_1.index[-1], type(DF_1.index[-1]))

    # if date is None:
    #     date = DF_1.index[-1].strftime('%Y-%m-%d')
    # S_1 = DF_1.loc[DF_1.index == date]
    # S_2 = DF_2.loc[DF_2.index == date]
    # DF = pd.DataFrame(data={MdlN: S_1.squeeze(), MdlN_B: S_2.squeeze()})
    # S_idx_upper = DF.index.to_series().astype(str).str.upper()
    # DF = DF.loc[
    #     ~(
    #         S_idx_upper.str.contains('IN-OUT', regex=False)
    #         | S_idx_upper.str.contains('PERCENT_DISCREPANCY', regex=False)
    #     )
    # ]

    # if sum_Pkg:
    #     S_idx = DF.index.to_series().astype(str)
    #     S_suffix = np.where(
    #         S_idx.str.endswith('_IN'),
    #         '_IN',
    #         np.where(S_idx.str.endswith('_OUT'), '_OUT', '_OTH'),
    #     )
    #     S_grp = S_idx.str[:3] + S_suffix
    #     DF = DF.groupby(S_grp, sort=False).sum(min_count=1)

    # # Add NET rows per parameter: NET = IN - OUT
    # S_idx = DF.index.to_series().astype(str)
    # DF_IN = DF.loc[S_idx.str.endswith('_IN')].copy()
    # DF_OUT = DF.loc[S_idx.str.endswith('_OUT')].copy()
    # if not DF_IN.empty or not DF_OUT.empty:
    #     DF_IN.index = DF_IN.index.to_series().str.replace('_IN$', '', regex=True)
    #     DF_OUT.index = DF_OUT.index.to_series().str.replace('_OUT$', '', regex=True)
    #     idx = DF_IN.index.union(DF_OUT.index)
    #     DF_NET = DF_IN.reindex(idx, fill_value=0).sub(DF_OUT.reindex(idx, fill_value=0), fill_value=0)
    #     DF_NET.index = DF_NET.index + '_NET'
    #     DF = pd.concat([DF, DF_NET], axis=0)

    # sorted_i = (
    #     [col for col in DF.index if str(col).endswith('_IN')]
    #     + [col for col in DF.index if str(col).endswith('_OUT')]
    #     + [col for col in DF.index if str(col).endswith('_NET')]
    #     + [col for col in DF.index if not str(col).endswith(('_IN', '_OUT', '_NET'))]
    # )
    # DF = DF.reindex(index=sorted_i)

    # DF['Diff'] = DF[MdlN] - DF[MdlN_B]
    # DF = DF.replace([np.inf, -np.inf], np.nan).round(0).astype('Int64')
    # DF['Diff_%'] = DF.apply(
    #     lambda x: x['Diff'] / x[MdlN_B] * 100 if pd.notnull(x['Diff']) and x[MdlN_B] != 0 else np.nan, axis=1
    # )
    # # Replace infinities, convert to nullable Int64, and display missing values as '-'
    # DF = DF.replace([np.inf, -np.inf], np.nan).round(0).astype('Int64')
    # DF.style.format(na_rep='-')
    # if Pa_Out is None:
    #     Pa_Out = M.Pa.PoP_Out_MdlN / f'WB_Diff_{MdlN}_vs_{MdlN_B}_{date}.xlsx'

    # Pa_Out = Path(Pa_Out)
    # Pa_Out.parent.mkdir(parents=True, exist_ok=True)
    # DF.to_excel(Pa_Out, index=True, na_rep='-')
    # sprint(f'🟢🟢🟢 - Saved WB stage to {Pa_Out}.')
