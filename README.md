# Introduction
This project folder is used for the purposes of [WaterScape](https://waterscape.sites.uu.nl/) Regional Groundwater modelling.<br>
---

# Knowledge requirements
- You need to be familiar with Groundwater Modelling concepts, version control/GitHub... TBC
- For any questions, contact m.karampasis@uu.nl
- **.\Mng\Acronyms.xlsx** contains acronyms and rules used throughout the project. It also describes the **naming convention** used in this project. You need to be familiar with the principles described in this file when working with this folder/project. New acronyms should be registered there. Make sure you read the instructions sheet/tab to understand the acronym system, and follow the system's rules when adding new shortcuts/abbreviations.
---

# Project set-up
- Read the entire **README.md** before working on the project.<br>
- Some project software require **Windows** to run, so unfortunately it's only possible to work on this project from a Win machine.
- Follow the following guide to set-up the project Env: **.\code\Env\how_to_make_Env.md**
	If you're using SURF, create a workspace as per the guide below. Select catalogue item "Win_TOTP_git_iBridges_pixi" when creating a workspace, as it'll give you acces to a lot of tools you need (git, pixi, iBridges)
		https://utrechtuniversity.github.io/vre-docs/docs/first-steps.html
		Then connect to the workspace.
- You'll get the model code and the configuration files via git when you follow the guide above, but the model files are only shared upon request. 
---

# Rules and Tips
- The folder is version controlled through **GitHub** (smaller files, e.g. **code**) and **DVC**. You can use "`git ls-files`" to print all Git tracked files, and "`dvc list . --dvc-only --recursive`" (warning, it's very slow), to print the files tracked with each of the two methods. Make sure you git push at frequent intervals.
- For good **data management**, it's advised to include **metadata** in data folders, where the origin of the data isn't self-explanatory. E.g. a README file in a folder with IDFs (IDFs may contain spatial data, units etc. but oftentimes their origin/method of production, which is usually needed to understand what they are, is missing from the metadata).
---

# Folder structure/description
Files that are specific to one of the **models** will be contained withing the folder of that model. The rest of the folders in this directory should contain files that are (or will be) used by/for multiple models.<br>
Below is a brief description of the contents of each main folder in this directory.<br>
- **code**: Contains scripts and code. Sub-folders grouped by function/purpose.
- **data**: Contains data not specific to one model (e.g. KNMI climate TS).
- **Mng**: Contains files used for managing the project. WS_RunLog.xlsx is used to keep track of runs. It reads log.csv, where Sim info is recorded as the Sim is being executed.
- **models**: Contains a sub-folder for each model used in the project. All data/files related to just one of the models belong inside those sub-folders. If no metadata about the file source is provided, assume the source is the Deltares P: drive (e.g. for NBr: under p:\archivedprojects\11209224-sponswerkingnl\ or p:\archivedprojects\11206534-002-imod-brabantsedelta\)<br>
the models folder structure is described in more detail below because it is complex and critical for the project.
- other: Files relevant to the project that don't belong in any of the other categories/folders.
- **software**: Contains modelling software
- SS: **Superseded** - anything not relevant anymore. (although superseded files can be found in other sub-folders too).
---

## Model(s)
All model sub-folders contain the same folder structure for consistency. Files in those folders are only relevant to this Mdl. The Fo Str is described below:
- **code**:	Contains code specific to each Mdl. e.g. Mdl_Prep contains the .bat & .ini file to prepare a Mdl run.
- **doc**:	self-explanatory
- **In**:	model inputs, organized by iMOD package/module (organized by type). Raw data shouldn't be stored here. It’s acceptable to have MtDt files and/or rasters here (for ease of use). It's even ok to have simple code too, but if it's routine code it needs to be moved somewhere more central (./code of ./models/*/code).
- **PoP**:	Contains post-processed (PoPed) files. More specifically:
	**common**:	general information layers, e.g. rivers, background map etc. - Those elements are used in the MM regardless of run.
	**In**:		files converted to GIS layers for review and visualization. Most inputs remain the same as the B for a run, but some have to be re-referenced on each Sim. This is done by changing the data-source of the layer (programmatically).
	**Clc_In**:	shapefiles created from In Calcs. e.g. transmissivity = K * b (thickness)
	**Out**:	contains a model map for each Sim. Post processed output (PoPed Out) is unique to each Sim. Thus, the baseline MM gets copied and the output layers, which are relatively referenced, automatically get linked to the new Sim's PoPed Out.
- **PrP**:	**Pre-Processing**. Can include raw or intermediate data, or even scripts/JupNotes to create the In files.
- **Sim**:	**Simulation** folders. 1 for each Sim. Unfortunately, the way iMOD is designed, Mdl Output needs to be saved here. Organized by Sim.
---

## Terminal tools
There is a list of **terminal tools** that facilitate common tasks for this project. Those are listed in G:/code/setup.py, with a brief description.<br>
To add another terminal command, you need to add it to the setup file (similar to the other commands), and make a script. Then you need to install WS_Mdl (as explained in the Python Env installation guide above).<br>
It's also possible to run Python functions from G:\code\WS_Mdl\ modules via:
WS_Mdl <function> <arg1> <arg2>
or if the function is not exposed via __all__ = [...] (in the same module), this might work:
"`WS_Mdl.module` <function> <arg1> <arg2> ...".

<args> can be regular terminal commands, e.g. NBr100, or kwargs, e.g. verbose=True, Pkgs=['DRN', 'RIV'] etc.
---