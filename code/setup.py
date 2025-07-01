from setuptools import setup, find_packages

setup(
    name="WS_Mdl",
    version="0.1",
    packages=find_packages(),
    # install_requires=[ # I think those aren't necessary as they're already in the conda environment.
    #     "pandas",
    #     "numpy",
    # ],
    entry_points={
        "console_scripts": [
            "WS_Mdl=WS_Mdl.__main__:main",  # Keep the main command
            "S_from_B=WS_Mdl.scripts.S_from_B:main", # Create Run Cfg Fis for S from B, then open them with the default editor.
            "S_from_B_undo=WS_Mdl.scripts.S_from_B_undo:main", # Deletes the Run Cfg Fis for a MdlN.
            "Dir_Fo_size=WS_Mdl.scripts.Dir_Fo_size:main",  # Gets all directory sizes
            "map_DVC=WS_Mdl.scripts.map_DVC:main",  # Maps all DVC'd files/directories
            "map_gitignore=WS_Mdl.scripts.map_gitignore:main",  # Maps all DVC'd files/directories
            "DVC_add_pattern=WS_Mdl.scripts.DVC_add_pattern:main", # Runs DVC add for all files directly under provided directory.
            "DVC_add_pattern_deep=WS_Mdl.scripts.DVC_add_pattern_deep:main", # Runs DVC add for all files directly under provided directory.
            "reset_Sim=WS_Mdl.scripts.reset_Sim:main", # Resets all Sims to a pre-run state (i.e. .bat, .ini. prj, .smk files are preserved, but MdlN Fo in Sim gets deleted). SHOULD ONLY be used in development stage. Either modify this, or make another function for archiving.
            "RunMng=WS_Mdl.scripts.RunMng:main", # Read the RunLog, and for each queued model, run the corresponding Snakemake file.
            "IDF_to_TIF=WS_Mdl.scripts.IDF_to_TIF:main", # Converts IDF to TIF using the provided IDF file.
            "Sim_Cfg=WS_Mdl.scripts.Sim_Cfg:main", # Opens SimCfg files for the provided MdlN.
            "open_LST=WS_Mdl.scripts.open_LST:main", # Opens LST files for the provided MdlN.
            "open_LSTs=WS_Mdl.scripts.open_LSTs:main", # Opens LST files for the provided MdlN.
            "rerun_Sim=WS_Mdl.scripts.rerun_Sim:main", # Reruns the Sim for the provided MdlN.
        ]
    },
)
