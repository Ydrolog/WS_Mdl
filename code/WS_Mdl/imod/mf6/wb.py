def WB_Diff_to_xlsx(
    MdlN: str, MdlN_B: str, date: str, sum_Pkg: bool = False, Pa_Out: str | None = None
):  # 666 add option to do cumulative of just for that SP
    """
    Compares the water budget of two models (MdlN and MdlN_B) for a specific date and saves the differences to an Excel file.
    - date: YYYY-MM-DD
    - sum_Pkg: if True, sum rows where the first 3 index characters match
    """

    from pathlib import Path

    from WS_Mdl.core.mdl import Mdl_N

    # Load basics
    set_verbose(False)
    M = Mdl_N(MdlN)
    M_B = Mdl_N(MdlN_B)
    d_INI = INI_to_d(M.Pa.INI)
    SP_date_1st = DT.strftime(DT.strptime(d_INI['SDATE'], '%Y%m%d'), '%Y-%m-%d')
    set_verbose(True)

    # Load budget to dataframes. fp.utils.Mf6ListBudget returns a tuple. 1st item is WB for each SP. 2nd item is cumulative.
    DF_1, DF_1_Tot = fp.utils.Mf6ListBudget(M.Pa.LST_Mdl).get_dataframes()
    DF_2, DF_2_Tot = fp.utils.Mf6ListBudget(M_B.Pa.LST_Mdl).get_dataframes()

    start_date = pd.to_datetime(SP_date_1st)
    for DF in [DF_1, DF_1_Tot, DF_2, DF_2_Tot]:
        DF.index = pd.date_range(start=start_date, periods=len(DF), freq='D')
    S_1 = DF_1_Tot.loc[DF_1_Tot.index == date]
    S_2 = DF_2_Tot.loc[DF_2_Tot.index == date]
    DF = pd.DataFrame(data={MdlN: S_1.squeeze(), MdlN_B: S_2.squeeze()})
    S_idx_upper = DF.index.to_series().astype(str).str.upper()
    DF = DF.loc[
        ~(
            S_idx_upper.str.contains('IN-OUT', regex=False)
            | S_idx_upper.str.contains('PERCENT_DISCREPANCY', regex=False)
        )
    ]
    sorted_i = (
        [col for col in DF.index if '_IN' in col]
        + [col for col in DF.index if '_OUT' in col]
        + [col for col in DF.index if '_OUT' not in col and '_IN' not in col]
    )
    DF = DF.reindex(index=sorted_i)

    if sum_Pkg:
        S_idx = DF.index.to_series().astype(str)
        S_suffix = np.where(
            S_idx.str.endswith('_IN'),
            '_IN',
            np.where(S_idx.str.endswith('_OUT'), '_OUT', '_OTH'),
        )
        S_grp = S_idx.str[:3] + S_suffix
        DF = DF.groupby(S_grp, sort=False).sum(min_count=1)

    DF['Diff'] = DF[MdlN] - DF[MdlN_B]
    DF = DF.replace([np.inf, -np.inf], np.nan).round(0).astype('Int64')
    DF['Diff_%'] = DF.apply(
        lambda x: x['Diff'] / x[MdlN_B] * 100 if pd.notnull(x['Diff']) and x[MdlN_B] != 0 else np.nan, axis=1
    )
    # Replace infinities, convert to nullable Int64, and display missing values as '-'
    DF = DF.replace([np.inf, -np.inf], np.nan).round(0).astype('Int64')
    DF.style.format(na_rep='-')
    if Pa_Out is None:
        Pa_Out = M.Pa.PoP_Out_MdlN / f'WB_Diff_{MdlN}_vs_{MdlN_B}_{date}.xlsx'

    Pa_Out = Path(Pa_Out)
    Pa_Out.parent.mkdir(parents=True, exist_ok=True)
    DF.to_excel(Pa_Out, index=True, na_rep='-')
    sprint(f'🟢🟢🟢 - Saved WB stage to {Pa_Out}.')
