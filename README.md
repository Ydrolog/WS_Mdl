## Introduction
This folder is used for the purposes of WaterScape Regional Groundwater modelling.<br>
From this point on, abbreviations/acronyms defined in .\Mng\Acronyms.xlsx are used (to save time and space).<br>
Initially, this folder is being set up in my (Marios Karampasis) OneDrive. To avoid potential errors caused by OneDrive's default name (includes a hyphen and spaces), make a permanent symbolic link on your C: drive using the following command in your CMD:<br>
mklink /D C:\OD "C:\Users\User\OneDrive - Universiteit Utrecht"<br>
(change the OneDrive directory to whatever it needs to be)<br>

The symbolic link can be deleted using:<br>
rmdir C:\OD<br>

All files you need to run the models should be available within this project folder. The only thing you need to install yourself is the python environment - use the guide below.<br>


## Rules and Tips
- .\Mng\Acronyms.xlsx contains acronyms and rules used throughout the project.  This is not just an acronym archive. It also explains the naming convention used throughout this project, so you need to be familiar with it's principles when working on with this folder/project. New acronyms should be registerred there. Make sure you read the instructions sheet/tab to understand the acronym system, and follow the system's rules when adding new shortcuts.
- The folder is version controlled through GitHub (and DVC). Make sure you push at frequent intervals. (#666 this needs to be developped)
- For good data management, it's advised to include meta-data in data folder that are not self-explanatory. E.g. a read-me file in a folder with IDFs (IDFs may contain spatial data, units etc. but oftentimes their origin/method of production, which can be very useful, is missing from the meta-data).


## Folder structure/description
Files that are specific to one of the models will be contained withing the folder of that model. The rest of the folders in this directory should contain files that are (or will be) used by/for multiple models.<br>
Below is a brief description of what is contained within each of the main folders of this directory.<br>
- code: Contains scripts and code. Sub-folders grouped by function/purpose.
- data: Contains data not specific to one model (e.g. KNMI climate TS).
- Mng: Contains files used for managing the project.
- models: Contains a sub-folder for each model used in the project. All data/files related to just one of the models belong inside those sub-folders. If no metadata about the file source is provided, asume the source is the Deltares P: drive (e.g. for NBr: under p:\archivedprojects\11209224-sponswerkingnl\ or p:\archivedprojects\11206534-002-imod-brabantsedelta\)<br>
the models folder structure is described in more detail below because it is complex and critical for the project.
- other: Files relevant to the project that don't belong in any of the other categories/folders.
- software: Contains modelling software 
- SS: Superseded - anything not relevant anymore. (although superseded files can be found in other sub-folders too.

** models **
All model folders contain the same Fo Str for consistency. Files in those folders are only relevant to this Mdl. The Fo Str is described below:
- code: Contains code specific to each Mdl. e.g. Mdl_Prep contains the .bat & .ini file to prepare a Mdl run.
- doc: self explanatory
- In: model inputs, organised by iMOD package/module (organised by type). Only contains files that are used directly in the model - i.e. no raw data etc.
- MM: Contains 3 types of elements:
- general information layers, e.g. rivers, background map etc. - Those elements are used in the MM regardless of run.
- In files converted to GIS layers for review and visualization. Most inputs remain the same as the B for a run, but some have to be re-referenced on each Sim. This is done by changining the data-source of the layer (programmatically).
- PoP: contains a model map for each Sim. PoPed output is unique to each Sim. Thus, the B MM gets copied and the output layers, which are relatively referenced, automatically get linked to the new Sim's PoPed Out.
- PrP: Pre-Processing. Can include raw or intermediate data, or even scripts/JupNotes to create the In files.
- Sim: Simulation folders. 1 for each Sim. Unfortunatelly, the way iMOD is designed, Mdl Output needs to be saved here. Organized by Sim.


## Python env installation guide
....................................
TO BE WRITTEN

for this project's python library:<br>
pip install -e C:\OD\WS_Mdl\code (or pip install -e C:\OD\WS_Mdl\code --use-pep517)

