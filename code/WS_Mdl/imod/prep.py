def Mdl_Prep(MdlN: str, Pa_MF6_DLL: str = None, Pa_MSW_DLL: str = None, verbose=False):
    """
    Prepares Sim Fis from In Fis.
    Ins need to be read and processed, then MF6 and MSW need to be coupled. Then Sim Ins can be written.
    """

    set_verbose(verbose)

    # Load paths and variables from PRJ & INI
    d_Pa = get_MdlN_Pa(MdlN)
    Pa_PRJ = d_Pa['PRJ']
    # Dir_PRJ = PDN(Pa_PRJ)
    d_INI = INI_to_d(d_Pa['INI'])
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    SP_date_1st, SP_date_last = [
        DT.strftime(DT.strptime(d_INI[f'{i}'], '%Y%m%d'), '%Y-%m-%d') for i in ['SDATE', 'EDATE']
    ]
    # dx = dy = float(d_INI['CELLSIZE'])

    if not Pa_MF6_DLL:  # If not specified, the default location will be used.
        Pa_MF6_DLL = d_Pa['MF6_DLL']
    if not Pa_MSW_DLL:
        Pa_MSW_DLL = d_Pa['MSW_DLL']

    # Load PRJ & regrid it to Mdl Aa
    PRJ_, _ = o_PRJ_with_OBS(Pa_PRJ)
    PRJ, period_data = PRJ_[0], PRJ_[1]
    PRJ_regrid = regrid_PRJ(
        PRJ, MdlN
    )  # Using original PRJ to load MF6 Mdl gives warnings (and it's very slow). Regridding works much better though.

    # Set outer boundaries to -1. Otherwise CHD won't be loaded properly.
    BND = PRJ_regrid['bnd']['ibound']
    BND.loc[:, [BND.y[0], BND.y[-1]], :] = -1  # Top and bottom rows
    BND.loc[:, :, [BND.x[0], BND.x[-1]]] = -1  # Left and right columns
    vprint('🟢 - Boundary conditions set successfully!')

    # Load MF6 Simulation
    times = pd.date_range(SP_date_1st, SP_date_last, freq='D')
    Sim_MF6 = mf6.Modflow6Simulation.from_imod5_data(
        PRJ_regrid, period_data, times
    )  # It can be further sped up by multi-processing, but this is not implemented yet.
    vprint('🟢 - MF6 Simulation loaded successfully!')
    # Sim_MF6[f'{MdlN}'] = Sim_MF6.pop('imported_model')  # Rename imported_model to MdlN.

    # Pass the Sim components to objects.
    MF6_Mdl = Sim_MF6['imported_model']
    MF6_Mdl['oc'] = mf6.OutputControl(save_head='last', save_budget='last')
    Sim_MF6['ims'] = mf6_solution_moderate_settings()  # Mimic iMOD5's "Moderate" settings.
    MF6_DIS = MF6_Mdl['dis']

    # Load MSW
    PRJ_MSW = {'cap': PRJ_regrid.copy()['cap'], 'extra': PRJ_regrid.copy()['extra']}  # Isolate MSW keys from PRJ.
    PRJ_MSW['extra']['paths'][2][0] = mete_grid_Cvt_to_AbsPa(
        Pa_PRJ, PRJ
    )  ## Fix mete_grid.inp relative paths. Replace the mete_grid.inp path in the PRJ_MSW dictionary
    MSW_Mdl = msw.MetaSwapModel.from_imod5_data(PRJ_MSW, MF6_DIS, times)  # Load MSW model from PRJ
    vprint('🟢 - MSW Simulation loaded successfully!')

    # Clip models
    Sim_MF6_AoI = Sim_MF6.clip_box(x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax)
    MF6_Mdl_AoI = Sim_MF6_AoI['imported_model']
    MSW_Mdl_AoI = MSW_Mdl.clip_box(
        x_min=Xmin, x_max=Xmax, y_min=Ymin, y_max=Ymax
    )  # clip_box doesn't clip the packages I clipped beforehand, but it clips non raster-like packages like WEL and removes packages that are not in the AoI.
    print(f'MF6 Model AoI DIS shape: {MF6_Mdl_AoI["dis"].dataset.sizes}')
    print(f'MSW Model AoI grid shape: {MSW_Mdl_AoI["grid"].dataset.sizes}')
    print('🟢 Both models successfully clipped to Area of Interest with compatible discretization!')

    ## I've sense checked that the AoI models are correct. Check imod_python_init_NBr32.ipynb for more info.

    # Load models into memory
    for pkg in MF6_Mdl_AoI.values():
        pkg.dataset.load()

    for pkg in MSW_Mdl_AoI.values():
        pkg.dataset.load()

    # Create mask from current regridded model (not the old one)
    mask = (
        MF6_Mdl_AoI.domain
    )  # 666 mask needs to be checked and potentially updated with -1 values at the edge of the Mdl Aa.
    Sim_MF6_AoI.mask_all_models(mask)
    DIS_AoI = MF6_Mdl_AoI['dis']

    ### MF6 cleanup
    try:
        for Pkg in [i for i in MF6_Mdl_AoI.keys() if ('riv' in i.lower()) or ('drn' in i.lower())]:
            MF6_Mdl_AoI[Pkg].cleanup(DIS_AoI)
    except Exception:
        print('Failed to cleanup packages. Proceeding without cleanup. Fingers crossed!')

    # MetaSWAP cleanup
    MSW_Mdl_AoI['grid'].dataset['rootzone_depth'] = MSW_Mdl_AoI['grid'].dataset['rootzone_depth'].fillna(1.0)

    # Coupling
    metamod_coupling = primod.MetaModDriverCoupling(
        mf6_model='imported_model', mf6_recharge_package='msw-rch', mf6_wel_package='msw-sprinkling'
    )
    metamod = primod.MetaMod(MSW_Mdl_AoI, Sim_MF6_AoI, coupling_list=[metamod_coupling])
    os.makedirs(d_Pa['Pa_MdlN'], exist_ok=True)  # Create simulation directory if it doesn't exist

    # Write Mdl Files
    metamod.write(
        directory=d_Pa['Pa_MdlN'],
        modflow6_dll=Pa_MF6_DLL,
        metaswap_dll=Pa_MSW_DLL,
        metaswap_dll_dependency=PDN(Pa_MF6_DLL),
    )

    # # Review execution times per cell
    try:
        result = sp.run(
            [d_Pa['coupler_Exe'], d_Pa['TOML']], cwd=d_Pa['Pa_MdlN'], capture_output=True, text=True, timeout=3600
        )  # 1 hour timeout

        print(f'Return code: {result.returncode}')
        if result.stdout:
            print('STDOUT:')
            print(result.stdout)
        if result.stderr:
            print('STDERR:')
            print(result.stderr)

        if result.returncode == 0:
            print('✅ Model execution completed successfully!')
        else:
            print(f'❌ Model execution failed with return code {result.returncode}')

    except sp.TimeoutExpired:
        print('⏰ Model execution timed out after 1 hour')
    except Exception as e:
        print(f'❌ Error executing model: {e}')
