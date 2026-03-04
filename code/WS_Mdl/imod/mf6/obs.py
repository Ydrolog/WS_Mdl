def add(MdlN: str, Opt: str = 'BEGIN OPTIONS\nEND OPTIONS', iMOD5=False):
    """
    Adds OBS file(s) from PRJ file OBS block to Mdl Sim (which iMOD can't do). Thus the OBS file needs to be written, and then a link to the OBS file needs to be created within the NAM file.
    Assumes OBS IPF file contains the following parameters/columns: 'Id', 'L', 'X', 'Y'
    for iMOD5 option check WS_Mdl.utils.MdlN_Pa() description.
    """

    sprint(pre_Sign)
    sprint('Running add_OBS ...')
    d_Pa = MdlN_Pa(MdlN, iMOD5=iMOD5)  # Get default directories
    Pa_MdlN, Pa_INI, Pa_PRJ = (
        d_Pa[k] for k in ['Pa_MdlN', 'INI', 'PRJ']
    )  # and pass them to objects that will be used in the function

    # Extract info from INI file.
    d_INI = INI_to_d(Pa_INI)
    Xmin, Ymin, Xmax, Ymax = [float(i) for i in d_INI['WINDOW'].split(',')]
    cellsize = float(d_INI['CELLSIZE'])
    # N_R, N_C = int( - (Ymin - Ymax) / cellsize ), int( (Xmax - Xmin) / cellsize ),

    # Read PRJ file to extract OBS block info - list of OBS files to be added.
    l_OBS_lines = r_PRJ_with_OBS(Pa_PRJ)[1]
    pattern = r"['\",]([^'\",]*?\.ipf)"  # Regex pattern to extract file paths ending in .ipf
    l_IPF = [
        match.group(1) for line in l_OBS_lines for match in re.finditer(pattern, line)
    ]  # Find all IPF files of the OBS block.

    # Iterate through OBS files of OBS blocks and add them to the Sim
    for i, path in enumerate(l_IPF):
        Pa_OBS_IPF = os.path.abspath(PJ(Pa_MdlN, path))  # path of IPF file. To be read.
        OBS_IPF_Fi = PBN(Pa_OBS_IPF)  # Filename of OBS file to be added to Sim (to be added without ending)
        if i == 0:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}.OBS6')  # path of OBS file. To be written.
        else:
            Pa_OBS = PJ(Pa_MdlN, f'GWF_1/MODELINPUT/{MdlN}_N{i}.OBS6')  # path of OBS file. To be written.

        DF_OBS_IPF = r_IPF_Spa(
            Pa_OBS_IPF
        )  # Get list of OBS items (without temporal dimension, as it's uneccessary for the OBS file, and takes ages to load)
        DF_OBS_IPF_MdlAa = DF_OBS_IPF.loc[
            ((DF_OBS_IPF['X'] > Xmin) & (DF_OBS_IPF['X'] < Xmax))
            & ((DF_OBS_IPF['Y'] > Ymin) & (DF_OBS_IPF['Y'] < Ymax))
        ].copy()  # Slice to OBS within the Mdl Aa (using INI window)

        DF_OBS_IPF_MdlAa['C'] = ((DF_OBS_IPF_MdlAa['X'] - Xmin) / cellsize).astype(
            np.int32
        ) + 1  # Calculate Cs. Xmin at the origin of the model.
        DF_OBS_IPF_MdlAa['R'] = (-(DF_OBS_IPF_MdlAa['Y'] - Ymax) / cellsize).astype(
            np.int32
        ) + 1  # Calculate Rs. Ymax at the origin of the model.

        DF_OBS_IPF_MdlAa.sort_values(
            by=['L', 'R', 'C'], ascending=[True, True, True], inplace=True
        )  # Let's sort the DF by L, R, C

        with open(Pa_OBS, 'w') as f:  # write OBS file(s)
            # sprint(Pa_MdlN, path, Pa_OBS_IPF, sep='\n')
            f.write(f'# created from {Pa_OBS_IPF}\n')
            f.write(Opt.encode().decode('unicode_escape'))  # write optional block
            f.write(f'\n\nBEGIN CONTINUOUS FILEOUT OBS_{OBS_IPF_Fi.split(".")[0]}.csv\n')

            for _, row in DF_OBS_IPF_MdlAa.drop_duplicates(subset=['Id', 'L', 'R', 'C']).iterrows():
                f.write(f' {row["Id"]} HEAD {row["L"]} {row["R"]} {row["C"]}\n')

            f.write('END CONTINUOUS\n')

        # Open NAM file and add OBS file to it
        lock = FL(d_Pa['NAM_Mdl'] + '.lock')  # Create a file lock to prevent concurrent writes
        with lock, open(d_Pa['NAM_Mdl'], 'r+') as f:
            l_NAM = f.read().split('END PACKAGES')
            f.seek(0)
            f.truncate()  # overwrite in-place
            Pa_OBS_Rel = os.path.relpath(Pa_OBS, Pa_MdlN)

            f.write(l_NAM[0])
            f.write(rf' OBS6 .\{Pa_OBS_Rel} OBS_{OBS_IPF_Fi.split(".")[0]}')
            f.write('\nEND PACKAGES')

            f.flush()
            os.fsync(f.fileno())  # ensure it’s on disk
            # lock is released automatically when the with-block closes
        sprint(f'🟢 - {Pa_OBS} has been added successfully!')
    sprint(post_Sign)
